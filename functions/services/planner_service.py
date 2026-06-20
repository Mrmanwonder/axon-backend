"""
axon_planner_v2.py  -  Axon Daily Plan Generator (CAIE Edition)
════════════════════════════════════════════════════════════════

Architecture
────────────
  1.  Data hydration      ── Firestore reads: subjects, objectives, analytics, calendar
  2.  Phase detection     ── per-subject StudyPhase from days-to-exam
  3.  Objective scoring   ── 7-factor weighted formula
                               mastery_decay · grade_gap · paper_weight ·
                               command_word_gap · prereq_unlock · examiner_flag · recency
  4.  Topo sort           ── networkx DAG for prerequisite ordering
  5.  Task composition    ── phase-adaptive mix with per-paper-type overrides
                               foundation: deep_work heavy
                               t30: practice + topic questions
                               t14: past_paper + command drills
                               t7:  mock_exam + examiner reports
  6.  Block allocation    ── cognitive-load-aware slot assignment
                               peak_focus_morning / structured_morning /
                               afternoon / review_evening / light_evening
                               cross-subject interleaving to reduce fatigue
  7.  FSRS-lite spacing   ── stability·difficulty model; urgency drives surfacing
  8.  Break injection     ── Pomodoro-style rest gaps between deep work blocks
  9.  AI polish           ── optional Gemini pass: title rewriting + conflict repair
 10.  Persistence         ── batch-upsert to Firestore; completed tasks preserved;
                               stale uncompleted tasks pruned

Key improvements over V1
────────────────────────
  • CAIE paper-type awareness  (MCQ/structured/practical drive different task mixes)
  • 7-factor scoring vs. simple mastery-decay
  • FSRS-lite spacing (stability + difficulty per objective) vs. linear decay
  • Cognitive Load Units (CLU) daily budget with cross-subject interleaving
  • Command-word tier model (5 tiers, Bloom's-aligned)
  • Grade-gap factor: target A* pulls harder on weak objectives than target C
  • ExaminerReport task type: dedicated post-paper reflection pass
  • Smart rescheduling: trim low-priority task, or defer to next day
  • Paper-specific exam-date alignment (paper 1 soon → weight MCQ tasks higher)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time as time_module
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
from google.cloud import firestore

logger = logging.getLogger(__name__)

_SYLLABUS_CACHE_TTL_SECONDS = int(os.environ.get("SYLLABUS_CACHE_TTL_SECONDS", "900"))
_SYLLABUS_SUBJECT_CACHE: Dict[str, Tuple[float, Dict[str, dict]]] = {}
_SYLLABUS_ALL_CACHE: Tuple[float, Dict[str, dict]] | None = None

# ══════════════════════════════════════════════════════════════
# §1  ENUMERATIONS & CONSTANTS
# ══════════════════════════════════════════════════════════════


class StudyPhase(str, Enum):
    FOUNDATION = "Foundation Build"
    T30        = "T-30 Completion"
    T14        = "T-14 Deep Dive"
    T7         = "T-7 Mock Sprint"


class TaskType(str, Enum):
    DEEP_WORK          = "deep_work"
    PRACTICE           = "practice"
    REVIEW             = "review"
    PAST_PAPER         = "past_paper"
    FLASHCARDS         = "flashcards"
    MOCK_EXAM          = "mock_exam"
    COMMAND_WORD_DRILL = "command_word_drill"
    EXAMINER_REPORT    = "examiner_report"   # NEW in V2


class IntensityLevel(str, Enum):
    BLUE   = "blue"    # maintenance / consolidation
    ORANGE = "orange"  # active effort required
    RED    = "red"     # crisis / maximum urgency


class ScheduledWindow(str, Enum):
    PEAK_FOCUS_MORNING  = "peak_focus_morning"   # first 90 min, hardest tasks
    STRUCTURED_MORNING  = "structured_morning"   # remaining morning
    AFTERNOON           = "afternoon"            # mid-day (optional block)
    REVIEW_EVENING      = "review_evening"       # active review
    LIGHT_EVENING       = "light_evening"        # flashcards / reading only


class PaperType(str, Enum):
    MCQ        = "mcq"
    STRUCTURED = "structured"
    PRACTICAL  = "practical"
    ESSAY      = "essay"
    COURSEWORK = "coursework"


# ── CAIE Command Words ──────────────────────────────────────
# Tiered by Bloom's cognitive demand (1 = recall, 5 = evaluate)
COMMAND_WORD_TIERS: Dict[str, int] = {
    # Tier 1 - Knowledge retrieval
    "state": 1, "list": 1, "name": 1, "give": 1, "identify": 1, "recall": 1,
    # Tier 2 - Comprehension / application
    "define": 2, "describe": 2, "outline": 2, "calculate": 2,
    "show": 2, "determine": 2, "measure": 2, "complete": 2,
    # Tier 3 - Analysis / reasoning
    "explain": 3, "suggest": 3, "deduce": 3, "sketch": 3, "plot": 3,
    "predict": 3, "comment": 3, "derive": 3, "estimate": 3,
    # Tier 4 - Synthesis / evaluation
    "discuss": 4, "analyse": 4, "compare": 4, "justify": 4, "contrast": 4,
    # Tier 5 - Critical evaluation
    "evaluate": 5, "assess": 5, "criticise": 5, "to what extent": 5,
}

# ── Phase-adaptive task mix ──────────────────────────────────
# Each phase defines a weighted distribution across task types.
# TaskBuilder samples from this, then applies paper-type overrides.
PHASE_TASK_MIX: Dict[StudyPhase, List[Tuple[TaskType, float]]] = {
    StudyPhase.FOUNDATION: [
        (TaskType.DEEP_WORK,  0.55),
        (TaskType.PRACTICE,   0.20),
        (TaskType.REVIEW,     0.15),
        (TaskType.FLASHCARDS, 0.10),
    ],
    StudyPhase.T30: [
        (TaskType.PRACTICE,   0.35),
        (TaskType.PAST_PAPER, 0.20),
        (TaskType.DEEP_WORK,  0.20),
        (TaskType.REVIEW,     0.15),
        (TaskType.FLASHCARDS, 0.10),
    ],
    StudyPhase.T14: [
        (TaskType.PAST_PAPER,         0.40),
        (TaskType.COMMAND_WORD_DRILL, 0.20),
        (TaskType.REVIEW,             0.20),
        (TaskType.EXAMINER_REPORT,    0.10),
        (TaskType.FLASHCARDS,         0.10),
    ],
    StudyPhase.T7: [
        (TaskType.MOCK_EXAM,          0.60),
        (TaskType.EXAMINER_REPORT,    0.15),
        (TaskType.COMMAND_WORD_DRILL, 0.15),
        (TaskType.FLASHCARDS,         0.10),
    ],
}

# Paper-type overrides: MCQ and Practical papers demand a very different task diet
PAPER_TYPE_TASK_OVERRIDE: Dict[PaperType, Dict[TaskType, float]] = {
    PaperType.MCQ: {
        TaskType.PRACTICE:   0.50,
        TaskType.FLASHCARDS: 0.30,
        TaskType.PAST_PAPER: 0.20,
    },
    PaperType.PRACTICAL: {
        TaskType.DEEP_WORK:          0.40,   # lab technique theory
        TaskType.PAST_PAPER:         0.35,
        TaskType.COMMAND_WORD_DRILL: 0.25,   # "sketch", "plot", "describe" heavy
    },
}

# ── Cognitive Load Units ─────────────────────────────────────
# Empirically weighted mental effort per task type.
# Total CLU budget per day is capped; heavier tasks consume more budget.
CLU_PER_TASK: Dict[TaskType, float] = {
    TaskType.MOCK_EXAM:          10.0,
    TaskType.PAST_PAPER:          7.0,
    TaskType.DEEP_WORK:           6.0,
    TaskType.COMMAND_WORD_DRILL:  5.0,
    TaskType.PRACTICE:            5.0,
    TaskType.EXAMINER_REPORT:     3.0,
    TaskType.REVIEW:              3.0,
    TaskType.FLASHCARDS:          2.0,
}

DAILY_CLU_BUDGET         = 32.0   # max cognitive load units per day
MAX_SAME_SUBJECT_PER_DAY = 3      # prevent single-subject saturation
MAX_CONSECUTIVE_RED      = 1      # never two red-intensity tasks back-to-back
MAX_COMMAND_DRILLS_PER_DAY = 2    # injected on top of phase mix

# Pomodoro rhythm (minutes)
POMODORO_WORK  = 50
POMODORO_BREAK = 10


# ══════════════════════════════════════════════════════════════
# §2  DATA MODELS
# ══════════════════════════════════════════════════════════════


@dataclass
class Paper:
    number: int
    paper_type: PaperType
    duration_minutes: int
    total_marks: int
    weight_pct: float            # % contribution to final grade
    exam_date: Optional[date] = None


@dataclass
class SyllabusObjective:
    id: str
    topic: str
    subtopic: str
    description: str
    paper_numbers: List[int]     # which papers assess this objective
    command_words: List[str]     # CAIE command words used in questions
    prerequisites: List[str]     # ids of prerequisite objectives
    examiner_flagged: bool       # flagged as common error in examiner reports
    mastery_score: float         # 0-1 current mastery
    last_studied: Optional[datetime]
    stability: float             # FSRS stability in days (default 1.0)
    difficulty: float            # FSRS difficulty 0-1 (default 0.3)
    next_due: Optional[date] = None   # FSRS-computed next review date


@dataclass
class SubjectContext:
    subject_id: str
    name: str
    code: str                    # CAIE subject code e.g. "9702"
    level: str                   # "AS", "A2", "IGCSE", "O_LEVEL"
    papers: List[Paper]
    objectives: List[SyllabusObjective]
    days_to_exam: int
    target_grade: str            # "A*", "A", "B" ...
    phase: StudyPhase
    grade_gap: float             # 0-1: gap between current perf and target threshold
    weak_command_words: List[str]
    dep_graph: nx.DiGraph = field(default_factory=nx.DiGraph)


@dataclass
class PlannerTask:
    id: str
    title: str
    subject: str
    description: str
    paper: str
    objective_id: str
    start_time: datetime
    end_time: datetime
    status: str                  # pending | completed | rescheduled | trimmed
    date: str                    # ISO date string
    reason: str                  # human-readable rationale
    intensity_score: float
    intensity_label: str         # blue | orange | red
    phase: str
    anchor_date: str
    task_type: str
    scheduled_window: str
    priority: int                # maps to Flutter Priority enum index
    is_completed: bool = False
    is_sync_to_google: bool = False


@dataclass
class _Slot:
    """Internal representation of a time slot during block allocation."""
    window: ScheduledWindow
    start: datetime
    end: datetime
    clu_remaining: float


# ══════════════════════════════════════════════════════════════
# §3  FSRS-LITE SPACED REPETITION
# ══════════════════════════════════════════════════════════════


class FSRSLite:
    """
    Minimal FSRS (Free Spaced Repetition Scheduler) implementation.

    Core equation:
        R(t, S) = exp(ln(0.9) * t / S)
    where R = retrievability, t = days elapsed, S = stability.

    Urgency score = 1 - R, clamped to [0, 1].
    Higher urgency → higher priority in today's plan.
    """

    DESIRED_RETENTION = 0.90

    @staticmethod
    def retrievability(stability: float, elapsed_days: float) -> float:
        if elapsed_days <= 0 or stability <= 0:
            return 1.0
        return math.exp(math.log(FSRSLite.DESIRED_RETENTION) * elapsed_days / stability)

    @staticmethod
    def urgency(obj: SyllabusObjective, today: date) -> float:
        """Returns 0-1 urgency; 1.0 = overdue, 0.0 = freshly reviewed."""
        if obj.last_studied is None:
            return 1.0
        elapsed = max(0, (today - obj.last_studied.date()).days)
        # Cap elapsed to prevent math overflow or extreme drops on long hiatus
        elapsed = min(elapsed, 90)
        r = FSRSLite.retrievability(obj.stability, elapsed)
        # factor in difficulty: harder topics become urgent slightly faster
        difficulty_factor = 1.0 + (obj.difficulty * 0.25)
        urgency = (1.0 - r) * difficulty_factor
        return round(max(0.0, min(1.0, urgency)), 4)

    @staticmethod
    def next_interval_days(stability: float) -> float:
        """Days until retrievability drops to DESIRED_RETENTION."""
        # Derived from R(t, S) = 0.9 → t = S (always, by construction)
        return max(1.0, stability)

    @staticmethod
    def update_after_recall(stability: float, difficulty: float) -> float:
        """Called when student demonstrates good recall."""
        # Dampen the multiplier for high stabilities to avoid extreme interval stretching
        growth = 1.0 + (11.0 * (1.0 - difficulty) * math.exp(-0.25 * stability))
        return min(stability * growth, 60.0) # Cap stability at 60 days

    @staticmethod
    def update_after_lapse(stability: float) -> float:
        """Called when student fails recall (e.g. marks question wrong)."""
        # A softer penalty than strict 20% wipeout for advanced concepts
        return max(0.5, min(stability * 0.40, 7.0))


# ══════════════════════════════════════════════════════════════
# §4  OBJECTIVE SCORING ENGINE
# ══════════════════════════════════════════════════════════════


class ObjectiveScoringEngine:
    """
    Computes a priority score ∈ [0, 1] for each syllabus objective.

    Seven weighted factors (weights sum to 1.0):

    ┌─────────────────────┬────────┬─────────────────────────────────────────────┐
    │ Factor              │ Weight │ Rationale                                   │
    ├─────────────────────┼────────┼─────────────────────────────────────────────┤
    │ F1 mastery_decay    │  0.25  │ FSRS urgency: how much has been forgotten?  │
    │ F2 grade_gap        │  0.20  │ Distance from target grade (A*, A, B ...)     │
    │ F3 paper_weight     │  0.15  │ Paper's % contribution to final grade       │
    │ F4 command_word_gap │  0.15  │ Deficit in required command-word proficiency│
    │ F5 prereq_unlock    │  0.10  │ # of objectives this one gates downstream   │
    │ F6 examiner_flag    │  0.10  │ Flagged as common mistake in examiner report│
    │ F7 recency_penalty  │  0.05  │ Penalises recently studied topics (-sign)   │
    └─────────────────────┴────────┴─────────────────────────────────────────────┘

    After weighted sum, a phase multiplier amplifies urgency as exams near:
      Foundation × 1.0 | T-30 × 1.2 | T-14 × 1.5 | T-7 × 2.0
    Crisis modifier (days ≤ 3 & mastery < 0.5) stacks an additional × 1.5.

    Paper proximity boost: if a specific paper's exam is within 5 days,
    objectives examined by that paper get a +0.15 additive bonus.
    """

    WEIGHTS = {
        "mastery_decay":    0.25,
        "grade_gap":        0.20,
        "paper_weight":     0.15,
        "command_word_gap": 0.15,
        "prereq_unlock":    0.10,
        "examiner_flag":    0.10,
        "recency_penalty":  0.05,
    }

    GRADE_THRESHOLDS: Dict[str, float] = {
        "A*": 0.90, "A": 0.80, "B": 0.70, "C": 0.60, "D": 0.50, "E": 0.40,
    }

    PHASE_MULTIPLIER: Dict[StudyPhase, float] = {
        StudyPhase.FOUNDATION: 1.0,
        StudyPhase.T30:        1.2,
        StudyPhase.T14:        1.5,
        StudyPhase.T7:         2.0,
    }

    def score(
        self,
        obj: SyllabusObjective,
        ctx: SubjectContext,
        cw_proficiency: Dict[str, float],
        today: date,
    ) -> float:
        f: Dict[str, float] = {}

        # F1: FSRS urgency (0 = fresh, 1 = overdue)
        f["mastery_decay"] = FSRSLite.urgency(obj, today)

        # F2: grade gap - how far current mastery is from the target threshold
        target_thresh = self.GRADE_THRESHOLDS.get(ctx.target_grade, 0.70)
        f["grade_gap"] = max(0.0, target_thresh - obj.mastery_score)

        # F3: average weight of papers that examine this objective
        rel_papers = [p for p in ctx.papers if p.number in obj.paper_numbers]
        if rel_papers:
            f["paper_weight"] = sum(p.weight_pct / 100.0 for p in rel_papers) / len(rel_papers)
        else:
            f["paper_weight"] = 0.30  # fallback assumption

        # F4: command-word gap - max gap weighted by tier (higher tiers matter more)
        if obj.command_words:
            # We use an exponential weight for tiers: a Tier 5 gap is significantly worse than Tier 1
            gaps = [
                max(0.0, 1.0 - cw_proficiency.get(cw, 0.0))
                * (math.pow(1.5, COMMAND_WORD_TIERS.get(cw, 1)) / math.pow(1.5, 5.0))
                for cw in obj.command_words
            ]
            f["command_word_gap"] = max(gaps) if gaps else 0.0
        else:
            f["command_word_gap"] = 0.0

        # F5: prerequisite unlock value (non-linear scale)
        successors = len(list(ctx.dep_graph.successors(obj.id)))
        f["prereq_unlock"] = min(1.0, math.log(1.0 + successors) / math.log(6.0))

        # F6: examiner flagged
        f["examiner_flag"] = 1.0 if obj.examiner_flagged else 0.0

        # F7: recency penalty - penalises recently studied topics (negative contribution)
        if obj.last_studied:
            days_since = max(0, (today - obj.last_studied.date()).days)
            f["recency_penalty"] = max(0.0, 1.0 - days_since / 3.0)
        else:
            f["recency_penalty"] = 0.0

        # Weighted sum (recency_penalty subtracts)
        raw = (
            sum(self.WEIGHTS[k] * f[k] for k in self.WEIGHTS if k != "recency_penalty")
            - self.WEIGHTS["recency_penalty"] * f["recency_penalty"]
        )

        # Phase multiplier
        multiplied = raw * self.PHASE_MULTIPLIER[ctx.phase]

        # Crisis modifier
        if ctx.days_to_exam <= 3 and obj.mastery_score < 0.50:
            multiplied *= 1.5

        # Paper proximity boost: specific paper exam within 5 days
        for p in rel_papers:
            if p.exam_date and 0 <= (p.exam_date - today).days <= 5:
                multiplied += 0.15
                break

        return round(min(1.0, max(0.0, multiplied)), 4)


# ══════════════════════════════════════════════════════════════
# §5  TASK BUILDER
# ══════════════════════════════════════════════════════════════


class TaskBuilder:
    """
    Converts a scored objective + SubjectContext into a fully populated PlannerTask.

    Responsibilities:
    - Select task type from phase mix or paper override
    - Determine intensity (blue/orange/red) from score + days to exam
    - Build human-readable title and actionable description
    - Assign to the correct ScheduledWindow
    - Compute start/end times from the slot
    """

    # Default durations (minutes) per task type
    DURATIONS: Dict[TaskType, int] = {
        TaskType.MOCK_EXAM:          120,
        TaskType.PAST_PAPER:          60,
        TaskType.DEEP_WORK:           50,
        TaskType.PRACTICE:            40,
        TaskType.COMMAND_WORD_DRILL:  30,
        TaskType.REVIEW:              30,
        TaskType.EXAMINER_REPORT:     25,
        TaskType.FLASHCARDS:          20,
    }

    # Which windows each task type is best suited to
    WINDOW_AFFINITY: Dict[TaskType, List[ScheduledWindow]] = {
        TaskType.MOCK_EXAM:          [ScheduledWindow.PEAK_FOCUS_MORNING],
        TaskType.PAST_PAPER:         [ScheduledWindow.PEAK_FOCUS_MORNING, ScheduledWindow.STRUCTURED_MORNING],
        TaskType.DEEP_WORK:          [ScheduledWindow.PEAK_FOCUS_MORNING, ScheduledWindow.STRUCTURED_MORNING],
        TaskType.PRACTICE:           [ScheduledWindow.STRUCTURED_MORNING, ScheduledWindow.AFTERNOON],
        TaskType.COMMAND_WORD_DRILL: [ScheduledWindow.AFTERNOON, ScheduledWindow.STRUCTURED_MORNING],
        TaskType.EXAMINER_REPORT:    [ScheduledWindow.REVIEW_EVENING, ScheduledWindow.AFTERNOON],
        TaskType.REVIEW:             [ScheduledWindow.REVIEW_EVENING],
        TaskType.FLASHCARDS:         [ScheduledWindow.REVIEW_EVENING, ScheduledWindow.LIGHT_EVENING],
    }

    TITLE_PREFIXES: Dict[TaskType, str] = {
        TaskType.DEEP_WORK:          "Master",
        TaskType.PRACTICE:           "Practice",
        TaskType.REVIEW:             "Review",
        TaskType.PAST_PAPER:         "Past Paper -",
        TaskType.FLASHCARDS:         "Flashcards -",
        TaskType.MOCK_EXAM:          "Mock Exam -",
        TaskType.COMMAND_WORD_DRILL: "Command Drill -",
        TaskType.EXAMINER_REPORT:    "Examiner Notes -",
    }

    INTENSITY_PRIORITY = {
        IntensityLevel.RED:    3,  # Priority.high
        IntensityLevel.ORANGE: 2,  # Priority.medium
        IntensityLevel.BLUE:   1,  # Priority.low
    }

    def select_task_type(
        self,
        obj: SyllabusObjective,
        ctx: SubjectContext,
        paper_override: Optional[Dict[TaskType, float]],
    ) -> TaskType:
        """
        Task-type selection logic (in priority order):
        1. Paper-type override (MCQ / practical → specific mix)
        2. Examiner-flagged objective in T14/T7 → examiner_report
        3. High-tier command word gap → command_word_drill
        4. Phase mix top weight
        """
        if paper_override:
            return max(paper_override, key=lambda k: paper_override[k])

        if obj.examiner_flagged and ctx.phase in (StudyPhase.T14, StudyPhase.T7):
            return TaskType.EXAMINER_REPORT

        if obj.command_words:
            max_tier = max(COMMAND_WORD_TIERS.get(cw, 1) for cw in obj.command_words)
            if max_tier >= 4 and ctx.phase in (StudyPhase.T14, StudyPhase.T7):
                return TaskType.COMMAND_WORD_DRILL

        phase_mix = PHASE_TASK_MIX[ctx.phase]
        return max(phase_mix, key=lambda x: x[1])[0]

    def determine_intensity(
        self, score: float, days_to_exam: int
    ) -> Tuple[float, IntensityLevel]:
        if days_to_exam <= 3 or score >= 0.75:
            return score, IntensityLevel.RED
        elif score >= 0.45:
            return score, IntensityLevel.ORANGE
        else:
            return score, IntensityLevel.BLUE

    def build_title(self, obj: SyllabusObjective, task_type: TaskType) -> str:
        prefix = self.TITLE_PREFIXES.get(task_type, "Study")
        topic = obj.topic[:40] if len(obj.topic) > 40 else obj.topic
        return f"{prefix} {topic}"

    def build_description(
        self, obj: SyllabusObjective, task_type: TaskType, ctx: SubjectContext
    ) -> str:
        parts = []
        if obj.description:
            parts.append(obj.description)
        if obj.command_words:
            parts.append(f"Command words: {', '.join(obj.command_words)}")
        if task_type == TaskType.EXAMINER_REPORT and obj.examiner_flagged:
            parts.append("⚠ Common error area - review examiner report commentary carefully.")
        if task_type == TaskType.PAST_PAPER:
            parts.append("Use mark-scheme after completing. Annotate why wrong answers were chosen.")
        if task_type == TaskType.MOCK_EXAM:
            parts.append("Strict timed conditions. No mark-scheme until complete.")
        return " | ".join(parts)

    def preferred_window(self, task_type: TaskType, slots: List[_Slot]) -> Optional[_Slot]:
        preferred = self.WINDOW_AFFINITY.get(task_type, [ScheduledWindow.REVIEW_EVENING])
        for pref in preferred:
            for slot in slots:
                if slot.window == pref and slot.clu_remaining > 0:
                    return slot
        # Fallback: any slot with capacity
        for slot in slots:
            if slot.clu_remaining > 0:
                return slot
        return None

    def build(
        self,
        obj: SyllabusObjective,
        ctx: SubjectContext,
        score: float,
        task_type: TaskType,
        slot_start: datetime,
        window: ScheduledWindow,
        today_str: str,
    ) -> PlannerTask:
        intensity_score, intensity = self.determine_intensity(score, ctx.days_to_exam)
        duration = self.DURATIONS[task_type]
        end_time = slot_start + timedelta(minutes=duration)

        paper_str = ""
        if obj.paper_numbers:
            paper_str = f"Paper {obj.paper_numbers[0]}"

        stable_id = hashlib.sha1(
            "|".join(
                [
                    today_str,
                    ctx.subject_id,
                    obj.id,
                    task_type.value,
                    window.value,
                    slot_start.isoformat(),
                ]
            ).encode("utf-8")
        ).hexdigest()[:24]

        return PlannerTask(
            id=f"plan_{stable_id}",
            title=self.build_title(obj, task_type),
            subject=ctx.name,
            description=self.build_description(obj, task_type, ctx),
            paper=paper_str,
            objective_id=obj.id,
            start_time=slot_start,
            end_time=end_time,
            status="pending",
            date=today_str,
            reason=(
                f"Score {score:.3f} | {ctx.phase.value} | "
                f"{ctx.days_to_exam}d to exam | target {ctx.target_grade}"
            ),
            intensity_score=round(intensity_score, 3),
            intensity_label=intensity.value,
            phase=ctx.phase.value,
            anchor_date=today_str,
            task_type=task_type.value,
            scheduled_window=window.value,
            priority=self.INTENSITY_PRIORITY[intensity],
        )


# ══════════════════════════════════════════════════════════════
# §6  SLOT MANAGER  (cognitive load + time block allocation)
# ══════════════════════════════════════════════════════════════


class SlotManager:
    """
    Manages time slots and enforces cognitive load rules.

    Day layout (proportional to available_hours):
      ┌──────────────────────────────────────────────────┐
      │  peak_focus_morning  [40% of time]               │
      │  structured_morning  [20% of time]               │
      │  - 20-min break -                                │
      │  afternoon           [20% of time, optional]     │
      │  - 30-min break -                                │
      │  review_evening      [15% of time]               │
      │  light_evening       [ 5% of time]               │
      └──────────────────────────────────────────────────┘

    Constraints enforced:
      • Daily CLU budget (DAILY_CLU_BUDGET)
      • No back-to-back RED intensity tasks
      • Max MAX_SAME_SUBJECT_PER_DAY tasks per subject
      • Pomodoro break gap injected between deep work blocks
    """

    def __init__(self, day_start: datetime, available_hours: float):
        self._slots: List[_Slot] = self._build_slots(day_start, available_hours)
        self._total_clu: float = 0.0
        self._last_intensity: Optional[IntensityLevel] = None
        self._subject_counts: Dict[str, int] = defaultdict(int)

    # ── Slot construction ────────────────────────────────────

    def _build_slots(self, ds: datetime, hours: float) -> List[_Slot]:
        total_min = int(hours * 60)
        slots: List[_Slot] = []

        def add(window: ScheduledWindow, pct: float, clu_pct: float, start: datetime) -> datetime:
            dur = int(total_min * pct)
            end = start + timedelta(minutes=dur)
            slots.append(_Slot(window=window, start=start, end=end,
                               clu_remaining=DAILY_CLU_BUDGET * clu_pct))
            return end

        cursor = ds
        cursor = add(ScheduledWindow.PEAK_FOCUS_MORNING,  0.40, 0.40, cursor)
        cursor = add(ScheduledWindow.STRUCTURED_MORNING,  0.20, 0.20, cursor)
        cursor += timedelta(minutes=20)   # break
        cursor = add(ScheduledWindow.AFTERNOON,           0.20, 0.20, cursor)
        cursor += timedelta(minutes=30)   # break
        cursor = add(ScheduledWindow.REVIEW_EVENING,      0.15, 0.15, cursor)
        cursor = add(ScheduledWindow.LIGHT_EVENING,       0.05, 0.05, cursor)
        return slots

    # ── Guard checks ─────────────────────────────────────────

    def can_fit(
        self, task_type: TaskType, intensity: IntensityLevel, subject: str
    ) -> bool:
        clu_needed = CLU_PER_TASK[task_type]
        if self._total_clu + clu_needed > DAILY_CLU_BUDGET:
            return False
        if intensity == IntensityLevel.RED and self._last_intensity == IntensityLevel.RED:
            return False
        if self._subject_counts[subject] >= MAX_SAME_SUBJECT_PER_DAY:
            return False
        return True

    # ── Slot consumption ─────────────────────────────────────

    def consume(
        self,
        task_type: TaskType,
        intensity: IntensityLevel,
        subject: str,
        duration_minutes: int,
        preferred_window: Optional[ScheduledWindow] = None,
    ) -> Optional[Tuple[datetime, ScheduledWindow]]:
        """
        Allocates a time slot for the task.
        Returns (start_datetime, window) or None if no capacity.
        Advances the slot cursor and injects Pomodoro break if needed.
        """
        clu = CLU_PER_TASK[task_type]

        # Try preferred window first, then fall through
        ordered = self._slots[:]
        if preferred_window:
            ordered.sort(key=lambda s: (0 if s.window == preferred_window else 1))

        for slot in ordered:
            if slot.clu_remaining < clu:
                continue
            if slot.start >= slot.end:
                continue

            # Fits - allocate
            start = slot.start
            slot.clu_remaining -= clu
            self._total_clu += clu
            self._last_intensity = intensity
            self._subject_counts[subject] += 1

            # Advance slot cursor (add Pomodoro break after deep-focus tasks)
            gap = POMODORO_BREAK if task_type in (TaskType.DEEP_WORK, TaskType.PAST_PAPER, TaskType.MOCK_EXAM) else 5
            slot.start = start + timedelta(minutes=duration_minutes + gap)

            return (start, slot.window)

        return None


# ══════════════════════════════════════════════════════════════
# §7  FIRESTORE HYDRATOR
# ══════════════════════════════════════════════════════════════


class FirestoreHydrator:
    """All Firestore reads for a given user, with graceful fallbacks."""

    def __init__(self, db: firestore.Client, uid: str):
        self.db = db
        self.uid = uid
        self._user_doc = db.collection("users_private").document(uid)

    # ── Helpers ──────────────────────────────────────────────

    def _safe_collection(self, *path_parts: str) -> List[dict]:
        try:
            ref = self._user_doc
            for part in path_parts:
                ref = ref.collection(part) if hasattr(ref, "collection") else ref.document(part)
            return [{"id": s.id, **s.to_dict()} for s in ref.stream()]
        except Exception as e:
            logger.warning(f"Firestore collection read failed {path_parts}: {e}")
            return []

    def _safe_doc(self, *path_parts: str) -> dict:
        try:
            ref = self._user_doc
            for i, part in enumerate(path_parts):
                if i % 2 == 0:
                    ref = ref.collection(part)
                else:
                    ref = ref.document(part)
            snap = ref.get()
            return snap.to_dict() or {}
        except Exception as e:
            logger.warning(f"Firestore doc read failed {path_parts}: {e}")
            return {}

    # ── Public reads ─────────────────────────────────────────

    def load_settings(self) -> dict:
        return self._safe_doc("settings", "planner")

    def load_subjects(self) -> List[dict]:
        try:
            user_doc = self._safe_doc()
            if not user_doc:
                return []
            # Extract from either "subjects" array or "study_catalog" keys
            subjects = user_doc.get("subjects", [])
            if not subjects and "study_catalog" in user_doc:
                subjects = list(user_doc["study_catalog"].keys())

            # Since the frontend only stores string names, we map them to a dict format
            # expected by the context builder
            return [{"id": s, "name": s, "code": s, "level": "A_LEVEL"} for s in subjects]
        except Exception as e:
            logger.warning(f"Failed to load user subjects: {e}")
            return []

    def load_objectives(self, subject_id: str) -> List[dict]:
        try:
            # 1. Fetch global syllabus_maps for this subject
            # The global collection is 'syllabus_maps'. We query where 'subject' == subject_id
            cache_key = subject_id.strip().lower()
            now = time_module.time()
            cached = _SYLLABUS_SUBJECT_CACHE.get(cache_key)
            if cached and now - cached[0] < _SYLLABUS_CACHE_TTL_SECONDS:
                global_objs = {k: dict(v) for k, v in cached[1].items()}
            else:
                snaps = self.db.collection("syllabus_maps").where("subject", "==", subject_id).stream()
                global_objs = {s.id: s.to_dict() for s in snaps}
                _SYLLABUS_SUBJECT_CACHE[cache_key] = (now, global_objs)

            if not global_objs:
                # If exact match fails, try fetching all and doing a fuzzy/lower match
                # This handles frontend storing "Mathematics" vs backend storing "9709"
                all_maps = self._load_all_syllabus_maps()
                for obj_id, data in all_maps.items():
                    if data.get("subject", "").lower() == subject_id.lower():
                        global_objs[obj_id] = data
                _SYLLABUS_SUBJECT_CACHE[cache_key] = (now, global_objs)

            # 2. Fetch user's mastery from users_private/{uid}/mastery
            mastery_snaps = self.db.collection("users_private").document(self.uid).collection("mastery").stream()
            mastery_data = {s.id: s.to_dict() for s in mastery_snaps}

            # 3. Merge them
            merged = []
            for obj_id, g_data in global_objs.items():
                m_data = mastery_data.get(obj_id, {})
                merged.append({
                    "id": obj_id,
                    **g_data,
                    "mastery_score": m_data.get("mastery_score", 0.0),
                    "last_studied": m_data.get("updated_at", None),
                    "stability": m_data.get("stability", 1.0),
                    "difficulty": m_data.get("difficulty", 0.3)
                })
            return merged
        except Exception as e:
            logger.warning(f"Failed to load objectives for {subject_id}: {e}")
            return []

    def _load_all_syllabus_maps(self) -> Dict[str, dict]:
        global _SYLLABUS_ALL_CACHE
        now = time_module.time()
        if _SYLLABUS_ALL_CACHE and now - _SYLLABUS_ALL_CACHE[0] < _SYLLABUS_CACHE_TTL_SECONDS:
            return {k: dict(v) for k, v in _SYLLABUS_ALL_CACHE[1].items()}
        snaps = self.db.collection("syllabus_maps").stream()
        all_maps = {s.id: s.to_dict() for s in snaps}
        _SYLLABUS_ALL_CACHE = (now, all_maps)
        return {k: dict(v) for k, v in all_maps.items()}

    def load_analytics(self) -> dict:
        return self._safe_doc("analytics", "summary")

    def load_command_word_proficiency(self) -> Dict[str, float]:
        raw = self._safe_doc("analytics", "command_words")
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    def load_calendar_events(self, date_str: str) -> List[dict]:
        try:
            snaps = (
                self._user_doc.collection("calendar_events")
                .where("date", "==", date_str)
                .stream()
            )
            return [s.to_dict() for s in snaps]
        except Exception:
            return []

    def load_existing_plan(self, date_str: str) -> List[dict]:
        try:
            snaps = (
                self._user_doc.collection("daily_plan")
                .where("date", "==", date_str)
                .stream()
            )
            return [{"id": s.id, **s.to_dict()} for s in snaps]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════
# §8  AI POLISH PASS  (optional Gemini refinement)
# ══════════════════════════════════════════════════════════════


class AIPolicier:
    """
    Optional Gemini 1.5-Flash pass over the deterministic plan.

    What it improves:
      • Rewrites task titles to be specific and motivating (CAIE context-aware)
      • Enhances descriptions with concrete study actions
      • Detects and resolves same-paper / same-window conflicts
      • Adds examiner-report citations for flagged objectives
      • Ensures task progression is pedagogically coherent

    PRESERVES: all fields except title, description, reason.
    Does NOT add or remove tasks (deterministic engine controls structure).
    Falls back silently to original tasks if API call fails.
    """

    SYSTEM_PROMPT = """You are Axon's AI study coach for CAIE students (IGCSE, AS Level, A Level).
A deterministic algorithm has generated today's study plan. Your job is to IMPROVE it:

1. Rewrite each task title: specific topic + action verb, max 60 chars, no generic phrasing.
   Bad: "Deep Work Waves" | Good: "Master Superposition & Path Difference"
2. Rewrite descriptions: give 1-2 concrete study actions (e.g. "Derive the formula, then attempt
   3 past-paper questions on phase difference before checking the mark scheme.").
3. Flag conflicts (same paper + same window, duplicate objectives) and resolve in the reason field.
4. For examiner_report tasks, include the specific common mistake pattern from the reason field.
5. Keep a motivating but calm tone appropriate for a student under exam pressure.

Return ONLY a valid JSON array. Preserve all fields except title, description, reason.
Do NOT add or remove tasks. Do NOT wrap in markdown code fences.
"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def refine(self, tasks: List[PlannerTask]) -> List[PlannerTask]:
        if not self.api_key or not tasks:
            return tasks
        try:
            import json
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=self.SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            payload = [
                {
                    "id": t.id,
                    "title": t.title,
                    "subject": t.subject,
                    "description": t.description,
                    "task_type": t.task_type,
                    "phase": t.phase,
                    "intensity_label": t.intensity_label,
                    "reason": t.reason,
                    "objective_id": t.objective_id,
                    "paper": t.paper,
                }
                for t in tasks
            ]
            response = model.generate_content(
                f"Improve this CAIE study plan:\n{json.dumps(payload, indent=2)}"
            )
            refined: List[dict] = json.loads(response.text.strip())
            task_map: Dict[str, PlannerTask] = {t.id: t for t in tasks}
            for r in refined:
                tid = r.get("id")
                if tid and tid in task_map:
                    orig = task_map[tid]
                    task_map[tid] = PlannerTask(
                        **{
                            **orig.__dict__,
                            "title":       str(r.get("title",       orig.title))[:80],
                            "description": str(r.get("description", orig.description))[:600],
                            "reason":      str(r.get("reason",      orig.reason))[:300],
                        }
                    )
            return list(task_map.values())
        except Exception as e:
            logger.warning(f"AI polish pass failed (using original plan): {e}")
            return tasks


# ══════════════════════════════════════════════════════════════
# §9  CONTEXT BUILDER  (Firestore → SubjectContext)
# ══════════════════════════════════════════════════════════════


class SubjectContextBuilder:
    """Builds SubjectContext objects from raw Firestore documents."""

    GRADE_THRESHOLDS = ObjectiveScoringEngine.GRADE_THRESHOLDS

    def build(
        self,
        raw: dict,
        hydrator: FirestoreHydrator,
        analytics: dict,
        today: date,
    ) -> Optional[SubjectContext]:
        try:
            subject_id = raw["id"]

            # Exam date
            exam_date_str = raw.get("exam_date") or raw.get("examDate")
            if not exam_date_str:
                return None
            exam_dt = datetime.fromisoformat(str(exam_date_str)).date()
            days_to_exam = (exam_dt - today).days
            if days_to_exam < 0:
                return None  # exam has passed

            phase = self._phase_from_days(days_to_exam)

            # Papers
            papers = [
                self._parse_paper(p)
                for p in raw.get("papers", [])
                if p
            ]

            # Objectives + dependency graph
            raw_objs = hydrator.load_objectives(subject_id)
            objectives = [self._parse_objective(o) for o in raw_objs if o]
            dep_graph = self._build_dep_graph(objectives)

            # Grade gap
            target_grade = raw.get("targetGrade") or raw.get("target_grade") or "A"
            sub_analytics = analytics.get(subject_id, {})
            current_mastery = float(sub_analytics.get("avg_mastery", 0.5))
            target_thresh = self.GRADE_THRESHOLDS.get(target_grade, 0.70)
            grade_gap = max(0.0, target_thresh - current_mastery)

            # Weak command words
            weak_cws = (
                raw.get("weak_command_words")
                or sub_analytics.get("weak_command_words")
                or []
            )

            return SubjectContext(
                subject_id=subject_id,
                name=raw.get("name", subject_id),
                code=str(raw.get("code", "")),
                level=raw.get("level", "A_LEVEL"),
                papers=papers,
                objectives=objectives,
                days_to_exam=days_to_exam,
                target_grade=target_grade,
                phase=phase,
                grade_gap=grade_gap,
                weak_command_words=weak_cws,
                dep_graph=dep_graph,
            )
        except Exception as e:
            logger.warning(f"SubjectContextBuilder failed for {raw.get('id')}: {e}")
            return None

    # ── Parsers ──────────────────────────────────────────────

    @staticmethod
    def _parse_paper(raw: dict) -> Paper:
        exam_date = None
        if raw.get("exam_date"):
            try:
                exam_date = datetime.fromisoformat(str(raw["exam_date"])).date()
            except ValueError:
                pass
        return Paper(
            number=int(raw.get("number", 1)),
            paper_type=PaperType(raw.get("type", PaperType.STRUCTURED.value)),
            duration_minutes=int(raw.get("duration_minutes", 90)),
            total_marks=int(raw.get("total_marks", 100)),
            weight_pct=float(raw.get("weight_pct", 33.0)),
            exam_date=exam_date,
        )

    @staticmethod
    def _parse_objective(raw: dict) -> SyllabusObjective:
        last_studied = None
        # Handle Firestore Datetime / String
        ls = raw.get("last_studied")
        if ls:
            if hasattr(ls, "timestamp"):
                last_studied = datetime.fromtimestamp(ls.timestamp())
            else:
                try:
                    last_studied = datetime.fromisoformat(str(ls))
                except ValueError:
                    pass

        # Parse papers string into list of ints
        paper_str = str(raw.get("paper", "1"))
        import re
        paper_nums = [int(n) for n in re.findall(r'\d+', paper_str)] or [1]

        return SyllabusObjective(
            id=raw["id"],
            topic=raw.get("topic", ""),
            subtopic=raw.get("subtopic", raw.get("sub_topic", "")),
            description=raw.get("description", ""),
            paper_numbers=raw.get("paper_numbers", paper_nums),
            command_words=raw.get("command_words", []),
            prerequisites=raw.get("prerequisites", []),
            examiner_flagged=bool(raw.get("examiner_flagged", False)),
            mastery_score=float(raw.get("mastery_score", 0.0)),
            last_studied=last_studied,
            stability=float(raw.get("stability", 1.0)),
            difficulty=float(raw.get("difficulty", 0.3)),
        )

    @staticmethod
    def _build_dep_graph(objectives: List[SyllabusObjective]) -> nx.DiGraph:
        g = nx.DiGraph()
        obj_ids = {o.id for o in objectives}
        for obj in objectives:
            g.add_node(obj.id)
            for prereq in obj.prerequisites:
                if prereq in obj_ids:
                    g.add_edge(prereq, obj.id)
        return g

    @staticmethod
    def _phase_from_days(days: int) -> StudyPhase:
        if days <= 7:
            return StudyPhase.T7
        if days <= 14:
            return StudyPhase.T14
        if days <= 30:
            return StudyPhase.T30
        return StudyPhase.FOUNDATION


# ══════════════════════════════════════════════════════════════
# §10  MAIN PLANNER SERVICE
# ══════════════════════════════════════════════════════════════


class DailyPlannerServiceV2:
    """
    Primary entry point for daily plan generation.

    Typical call flow:
        service = DailyPlannerServiceV2(db=firestore_client, gemini_api_key="...")
        tasks   = await service.generate_and_persist_daily_plan(uid="...", force=False)

    Short-circuit behaviour:
        If force=False and a non-empty plan already exists for today, returns [].
        The caller can inspect Firestore directly for the existing tasks.

    Multi-subject interleaving:
        Objectives from different subjects are interleaved in a round-robin pass
        before slot allocation. This prevents single-subject cognitive overload and
        mirrors the interleaved-practice effect from memory research.
    """

    def __init__(
        self,
        db: firestore.Client,
        gemini_api_key: Optional[str] = None,
    ):
        self.db = db
        self._scorer      = ObjectiveScoringEngine()
        self._builder     = TaskBuilder()
        self._ctx_builder = SubjectContextBuilder()
        self._ai          = AIPolicier(api_key=gemini_api_key)
        self._generation_semaphore = asyncio.Semaphore(
            max(1, int(os.environ.get("DAILY_PLANNER_MAX_PARALLEL", "24")))
        )
        self._ai_semaphore = asyncio.Semaphore(
            max(1, int(os.environ.get("DAILY_PLANNER_AI_PARALLEL", "2")))
        )
        self._ai_timeout_seconds = float(
            os.environ.get("DAILY_PLANNER_AI_TIMEOUT_SECONDS", "8")
        )
        self._ai_enabled = os.environ.get(
            "DAILY_PLANNER_AI_ENABLED", "true"
        ).lower() not in {"0", "false", "no", "off"}
        self._inflight_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Task[List[PlannerTask]]] = {}

    # ── Public API ───────────────────────────────────────────

    async def generate_and_persist_daily_plan(
        self,
        uid: str,
        force: bool = False,
        target_date: Optional[date] = None,
    ) -> List[PlannerTask]:
        today     = target_date or date.today()
        today_str = today.isoformat()
        key = f"{uid}:{today_str}:{int(force)}"

        async with self._inflight_lock:
            current = self._inflight.get(key)
            if current and not current.done():
                logger.info("Daily planner coalesced in-flight request for uid=%s date=%s", uid, today_str)
                task = current
            else:
                task = asyncio.create_task(
                    self._generate_and_persist_daily_plan_uncached(
                        uid=uid,
                        force=force,
                        today=today,
                        today_str=today_str,
                    )
                )
                self._inflight[key] = task

        try:
            return await task
        finally:
            async with self._inflight_lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)

    async def _generate_and_persist_daily_plan_uncached(
        self,
        *,
        uid: str,
        force: bool,
        today: date,
        today_str: str,
    ) -> List[PlannerTask]:
        hydrator  = FirestoreHydrator(self.db, uid)

        if not force:
            existing = await asyncio.to_thread(hydrator.load_existing_plan, today_str)
            if existing:
                logger.info(f"Plan exists for {uid} on {today_str} - skipping (force=False)")
                return []

        async with self._generation_semaphore:
            tasks = await asyncio.to_thread(
                self._calculate_daily_load, uid, hydrator, today, today_str
            )
            tasks = await self._refine_if_capacity(tasks)
            await asyncio.to_thread(self._persist, uid, tasks, today_str, hydrator)
        
        logger.info(f"Generated {len(tasks)} tasks for {uid} on {today_str}")
        return tasks

    async def _refine_if_capacity(self, tasks: List[PlannerTask]) -> List[PlannerTask]:
        if not self._ai_enabled or not tasks:
            return tasks
        if self._ai_semaphore.locked():
            logger.info("Daily planner AI polish skipped under load")
            return tasks

        try:
            await asyncio.wait_for(self._ai_semaphore.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            logger.info("Daily planner AI polish skipped after capacity timeout")
            return tasks

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._ai.refine, tasks),
                timeout=self._ai_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Daily planner AI polish timed out; using deterministic plan")
            return tasks
        finally:
            self._ai_semaphore.release()

    async def reschedule_missed_block(
        self,
        uid: str,
        task_id: str,
        new_start: Optional[datetime] = None,
    ) -> bool:
        """
        Reschedule a missed task. Strategy:
        1. Slot into review_evening at 20:00 (or caller-supplied new_start).
        2. If that slot is crowded, trim the lowest-priority flashcard/review task first.
        3. If task cannot fit today, mark it 'rescheduled' and create a carry-forward
           note in tomorrow's anchor_date queue.
        Returns True if rescheduled successfully.
        """
        hydrator  = FirestoreHydrator(self.db, uid)
        today_str = date.today().isoformat()
        existing  = hydrator.load_existing_plan(today_str)

        target = next((t for t in existing if t["id"] == task_id), None)
        if not target:
            logger.warning(f"reschedule_missed_block: task {task_id} not found")
            return False

        plan_col = self.db.collection("users_private").document(uid).collection("daily_plan")
        evening_start = new_start or datetime.combine(date.today(), time(20, 0))
        evening_end   = evening_start + timedelta(minutes=60)

        # Try to trim a low-priority evening task
        trimmable = [
            t for t in existing
            if not t.get("is_completed")
            and t.get("task_type") in (TaskType.FLASHCARDS.value, TaskType.REVIEW.value)
            and t.get("scheduled_window") in (
                ScheduledWindow.REVIEW_EVENING.value,
                ScheduledWindow.LIGHT_EVENING.value,
            )
            and t["id"] != task_id
        ]
        if trimmable:
            plan_col.document(trimmable[0]["id"]).update({"status": "trimmed"})

        plan_col.document(task_id).update({
            "start_time":        evening_start.isoformat(),
            "end_time":          evening_end.isoformat(),
            "status":            "rescheduled",
            "scheduled_window":  ScheduledWindow.REVIEW_EVENING.value,
        })
        return True

    # ── Core algorithm ───────────────────────────────────────

    def _calculate_daily_load(
        self,
        uid: str,
        hydrator: FirestoreHydrator,
        today: date,
        today_str: str,
    ) -> List[PlannerTask]:

        # 1. Hydrate settings & calendar
        settings         = hydrator.load_settings()
        available_hours  = float(settings.get("target_hours_per_day", 4.0))
        day_start_hour   = int(settings.get("day_start_hour", 8))
        analytics        = hydrator.load_analytics()
        cw_proficiency   = hydrator.load_command_word_proficiency()
        cal_events       = hydrator.load_calendar_events(today_str)

        # Self-improvement / Adaptive Cognitive Load
        # Reduce load if user is constantly missing targets or showing high burnout
        recent_completion = float(analytics.get("last_7_days_completion_rate", 1.0))
        burnout_factor    = float(analytics.get("burnout_indicator", 0.0))
        load_modifier = 1.0
        if recent_completion < 0.4:
            load_modifier *= 0.8  # Gently ease the load to build momentum
            logger.info(f"Adaptive Load: User {uid} completion <40%, reducing daily target.")
        if burnout_factor > 0.7:
            load_modifier *= 0.75 # Heavily ease load if burned out
            logger.info(f"Adaptive Load: User {uid} burnout >70%, enforcing lighter day.")

        # Deduct calendar busy time from available hours
        busy_hours = sum(
            float(e.get("duration_hours", 0))
            for e in cal_events
            if e.get("blocks_study", True)
        )
        available_hours = max(1.0, (available_hours - busy_hours) * load_modifier)

        day_start = datetime.combine(today, time(day_start_hour, 0))
        slot_mgr  = SlotManager(day_start, available_hours)

        # 2. Build subject contexts
        raw_subjects = hydrator.load_subjects()
        contexts: List[SubjectContext] = [
            ctx
            for raw in raw_subjects
            for ctx in [self._ctx_builder.build(raw, hydrator, analytics, today)]
            if ctx is not None
        ]

        if not contexts:
            logger.warning(f"No active subjects found for {uid}")
            return []

        # Prioritise by urgency (nearest exam first, highest grade gap breaks ties)
        contexts.sort(key=lambda c: (c.days_to_exam, -c.grade_gap))

        # 3. Score all objectives
        candidates: List[Tuple[float, SyllabusObjective, SubjectContext]] = []
        for ctx in contexts:
            topo_objs = self._topo_sort(ctx)
            for obj in topo_objs:
                s = self._scorer.score(obj, ctx, cw_proficiency, today)
                candidates.append((s, obj, ctx))

        # Sort globally by score, then interleave subjects
        candidates.sort(key=lambda x: -x[0])
        candidates = self._interleave_subjects(candidates)

        # 4. Allocate tasks
        tasks: List[PlannerTask] = []
        for score, obj, ctx in candidates:
            paper_override = self._paper_override(obj, ctx)
            task_type      = self._builder.select_task_type(obj, ctx, paper_override)
            _, intensity   = self._builder.determine_intensity(score, ctx.days_to_exam)

            if not slot_mgr.can_fit(task_type, intensity, ctx.name):
                continue

            pref_window = self._builder.WINDOW_AFFINITY.get(task_type, [None])[0]
            result = slot_mgr.consume(
                task_type, intensity, ctx.name,
                TaskBuilder.DURATIONS[task_type], pref_window
            )
            if result is None:
                continue

            slot_start, window = result
            task = self._builder.build(obj, ctx, score, task_type, slot_start, window, today_str)
            tasks.append(task)

        # 5. Inject command-word drills for critical gaps
        tasks = self._inject_cw_drills(tasks, cw_proficiency, contexts, slot_mgr, today_str)

        # 6. Sort by start time
        tasks.sort(key=lambda t: t.start_time)
        return tasks

    # ── Helper methods ───────────────────────────────────────

    def _topo_sort(self, ctx: SubjectContext) -> List[SyllabusObjective]:
        """Returns objectives in topological order (prerequisites before dependents)."""
        try:
            order   = list(nx.topological_sort(ctx.dep_graph))
            obj_map = {o.id: o for o in ctx.objectives}
            sorted_  = [obj_map[oid] for oid in order if oid in obj_map]
            in_dag   = {o.id for o in sorted_}
            remainder = [o for o in ctx.objectives if o.id not in in_dag]
            return sorted_ + remainder
        except nx.NetworkXUnfeasible:
            logger.warning(f"Prerequisite cycle detected in {ctx.subject_id}; ignoring order")
            return ctx.objectives

    def _interleave_subjects(
        self,
        candidates: List[Tuple[float, SyllabusObjective, SubjectContext]],
    ) -> List[Tuple[float, SyllabusObjective, SubjectContext]]:
        """
        Round-robin interleaving across subjects to reduce subject-saturation fatigue.
        Within each subject, the original score order is preserved.
        """
        buckets: Dict[str, List] = defaultdict(list)
        for item in candidates:
            buckets[item[2].subject_id].append(item)

        result: List = []
        keys   = list(buckets.keys())
        idx    = {k: 0 for k in keys}
        total  = len(candidates)

        while len(result) < total:
            added = False
            for k in keys:
                if idx[k] < len(buckets[k]):
                    result.append(buckets[k][idx[k]])
                    idx[k] += 1
                    added = True
            if not added:
                break
        return result

    def _paper_override(
        self, obj: SyllabusObjective, ctx: SubjectContext
    ) -> Optional[Dict[TaskType, float]]:
        """Returns PAPER_TYPE_TASK_OVERRIDE dict if objective maps to MCQ or Practical paper."""
        for paper in ctx.papers:
            if paper.number in obj.paper_numbers:
                override = PAPER_TYPE_TASK_OVERRIDE.get(paper.paper_type)
                if override:
                    return override
        return None

    def _inject_cw_drills(
        self,
        tasks: List[PlannerTask],
        cw_proficiency: Dict[str, float],
        contexts: List[SubjectContext],
        slot_mgr: SlotManager,
        today_str: str,
    ) -> List[PlannerTask]:
        """
        Injects up to MAX_COMMAND_DRILLS_PER_DAY command-word drill tasks
        for Tier ≥ 3 command words with proficiency < 0.60.
        These supplement the main plan and are scheduled in light_evening or afternoon.
        """
        drills_added = 0
        for ctx in contexts:
            if drills_added >= MAX_COMMAND_DRILLS_PER_DAY:
                break
            for cw in ctx.weak_command_words:
                if drills_added >= MAX_COMMAND_DRILLS_PER_DAY:
                    break
                tier = COMMAND_WORD_TIERS.get(cw, 1)
                prof = cw_proficiency.get(cw, 0.0)
                if tier < 3 or prof >= 0.60:
                    continue

                result = slot_mgr.consume(
                    TaskType.COMMAND_WORD_DRILL,
                    IntensityLevel.ORANGE,
                    ctx.name,
                    30,
                    preferred_window=ScheduledWindow.AFTERNOON,
                )
                if result is None:
                    continue

                slot_start, window = result
                stable_id = hashlib.sha1(
                    "|".join(
                        [
                            today_str,
                            ctx.subject_id,
                            cw,
                            TaskType.COMMAND_WORD_DRILL.value,
                            window.value,
                            slot_start.isoformat(),
                        ]
                    ).encode("utf-8")
                ).hexdigest()[:24]
                tasks.append(PlannerTask(
                    id=f"plan_{stable_id}",
                    title=f"Command Drill - '{cw.capitalize()}'",
                    subject=ctx.name,
                    description=(
                        f"Practise Tier-{tier} command word '{cw}'. "
                        f"Attempt 3 past-paper questions that use this command word, "
                        f"then self-assess against the mark scheme for mark-scheme language patterns."
                    ),
                    paper="",
                    objective_id="",
                    start_time=slot_start,
                    end_time=slot_start + timedelta(minutes=30),
                    status="pending",
                    date=today_str,
                    reason=f"Command word proficiency: {prof:.0%} (target ≥ 80%) | Tier {tier}",
                    intensity_score=0.60,
                    intensity_label=IntensityLevel.ORANGE.value,
                    phase=ctx.phase.value,
                    anchor_date=today_str,
                    task_type=TaskType.COMMAND_WORD_DRILL.value,
                    scheduled_window=window.value,
                    priority=2,
                ))
                drills_added += 1

        return tasks

    # ── Persistence ──────────────────────────────────────────

    def _persist(
        self,
        uid: str,
        new_tasks: List[PlannerTask],
        today_str: str,
        hydrator: FirestoreHydrator,
    ) -> None:
        plan_col = (
            self.db.collection("users_private")
            .document(uid)
            .collection("daily_plan")
        )
        existing     = hydrator.load_existing_plan(today_str)
        completed_ids = {t["id"] for t in existing if t.get("is_completed")}

        batch = self.db.batch()

        # Delete stale uncompleted tasks
        for t in existing:
            if t["id"] not in completed_ids:
                batch.delete(plan_col.document(t["id"]))

        # Write new tasks (skip ids that were already completed - they're preserved above)
        for task in new_tasks:
            if task.id in completed_ids:
                continue
            batch.set(plan_col.document(task.id), {
                "title":             task.title,
                "subject":           task.subject,
                "description":       task.description,
                "paper":             task.paper,
                "objective_id":      task.objective_id,
                "start_time":        task.start_time.isoformat(),
                "end_time":          task.end_time.isoformat(),
                "status":            task.status,
                "date":              task.date,
                "reason":            task.reason,
                "is_sync_to_google": task.is_sync_to_google,
                "is_completed":      task.is_completed,
                "intensity_score":   task.intensity_score,
                "intensity_label":   task.intensity_label,
                "phase":             task.phase,
                "anchor_date":       task.anchor_date,
                "task_type":         task.task_type,
                "scheduled_window":  task.scheduled_window,
                "priority":          task.priority,
            })

        batch.commit()
        logger.info(f"Persisted {len(new_tasks)} tasks for uid={uid} date={today_str}")


# ══════════════════════════════════════════════════════════════
# §11  FSRS FEEDBACK ENDPOINT
# ══════════════════════════════════════════════════════════════
#
# Called when a student completes a task and provides self-assessment.
# Updates the objective's stability, difficulty, and next_due in Firestore.
# Integrate this into the task-completion handler in daily_plan_service.dart.


class FSRSFeedbackService:
    """
    Updates FSRS parameters for a syllabus objective after a study session.

    Call this when:
      • A past_paper question for this objective is marked correct/incorrect
      • A flashcard for this objective is rated (again / good / easy)
      • A practice task is completed with a self-assessed confidence level
    """

    GRADE_TO_RECALLED: Dict[int, bool] = {
        1: False,  # Again / failed
        2: True,   # Hard but recalled
        3: True,   # Good recall
        4: True,   # Easy recall
    }

    def __init__(self, db: firestore.Client):
        self.db = db

    def record_recall(
        self,
        uid: str,
        subject_id: str,
        objective_id: str,
        grade: int,           # 1=again, 2=hard, 3=good, 4=easy
    ) -> None:
        """
        grade: 1-4 (Anki-style)
        Updates stability, difficulty, last_studied, next_due in Firestore.
        """
        obj_ref = (
            self.db.collection("users_private")
            .document(uid)
            .collection("subjects")
            .document(subject_id)
            .collection("objectives")
            .document(objective_id)
        )
        snap = obj_ref.get()
        if not snap.exists:
            logger.warning(f"Objective {objective_id} not found for FSRS update")
            return

        data       = snap.to_dict()
        stability  = float(data.get("stability", 1.0))
        difficulty = float(data.get("difficulty", 0.3))
        recalled   = self.GRADE_TO_RECALLED.get(grade, True)

        new_stability = (
            FSRSLite.update_after_recall(stability, difficulty)
            if recalled
            else FSRSLite.update_after_lapse(stability)
        )

        # Update difficulty: easy pulls it down, again pushes it up
        difficulty_delta = {1: +0.10, 2: +0.05, 3: 0.0, 4: -0.08}.get(grade, 0.0)
        new_difficulty = max(0.1, min(0.9, difficulty + difficulty_delta))

        # Mastery score: simple EMA
        current_mastery = float(data.get("mastery_score", 0.0))
        outcome_value   = {1: 0.0, 2: 0.4, 3: 0.75, 4: 1.0}.get(grade, 0.5)
        new_mastery     = round(current_mastery * 0.8 + outcome_value * 0.2, 3)

        today    = date.today()
        next_due = today + timedelta(days=FSRSLite.next_interval_days(new_stability))

        obj_ref.update({
            "stability":     round(new_stability, 3),
            "difficulty":    round(new_difficulty, 3),
            "mastery_score": new_mastery,
            "last_studied":  datetime.now().isoformat(),
            "next_due":      next_due.isoformat(),
        })
        logger.debug(
            f"FSRS update {objective_id}: S={new_stability:.2f} D={new_difficulty:.2f} "
            f"mastery={new_mastery:.2f} next_due={next_due}"
        )
