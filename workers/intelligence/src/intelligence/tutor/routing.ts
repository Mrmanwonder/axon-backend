import type { Intent, TutorRequest } from "../../schemas";

const currentPattern = /\b(current|latest|today|this year|20\d{2}|syllabus|exam date|admission|law|statistics|recent|software version)\b/i;
const calculationPattern = /(?:\d\s*[+\-*/^=]|solve\s+|calculate\s+|simplif|factor|equivalent)/i;
const unitsPattern = /\b(unit|convert|metres?|meters?|kilograms?|seconds?|newtons?|joules?|watts?|volts?)\b/i;
const chemistryPattern = /\b(balance|molar mass|stoichiometr|oxidation|chemical equation|moles?)\b/i;

export interface ToolDecision {
  retrieval: boolean;
  calculator: boolean;
  symbolicMath: boolean;
  units: boolean;
  chemistry: boolean;
  paperEvidence: boolean;
}

export function detectIntent(request: TutorRequest): Intent {
  const text = request.message.toLowerCase();
  if (/\b(hint|nudge)\b/.test(text)) return "hint";
  if (currentPattern.test(text)) return "current_information";
  if (/\b(why.*marks?|teacher.*marks?|paper feedback|marks? lost)\b/.test(text)) return "paper_feedback";
  if (/\b(check my|is my work|is this correct)\b/.test(text)) return "work_check";
  if (/\b(where did i go wrong|my mistake|what did i do wrong)\b/.test(text)) return "mistake_diagnosis";
  if (/\b(compare|difference between)\b/.test(text)) return "comparison";
  if (/\b(quiz me|ask me|socratic)\b/.test(text)) return "socratic";
  if (calculationPattern.test(text)) return "problem_solving";
  if (/\b(source|citation|according to)\b/.test(text)) return "source_question";
  if (/\b(explain|teach|why|how)\b/.test(text)) return "concept_explanation";
  if (text.length < 40 && /^(hi|hello|thanks|thank you)/.test(text)) return "casual";
  return "direct_answer";
}

export function decideTools(request: TutorRequest, intent: Intent): ToolDecision {
  return {
    retrieval: intent === "current_information" || intent === "source_question",
    calculator: intent === "problem_solving" && /\d/.test(request.message),
    symbolicMath: intent === "problem_solving" && calculationPattern.test(request.message),
    units: unitsPattern.test(request.message),
    chemistry: chemistryPattern.test(request.message),
    paperEvidence: intent === "paper_feedback" || intent === "work_check" || intent === "mistake_diagnosis"
  };
}

export function promptIdFor(intent: Intent): string {
  switch (intent) {
    case "hint": return "tutor.hint.v2";
    case "paper_feedback": return "tutor.paper_explanation.v3";
    case "mistake_diagnosis": return "tutor.mistake_diagnosis.v2";
    case "comparison": return "tutor.compare_working.v1";
    case "socratic": return "tutor.socratic.v2";
    case "current_information": case "source_question": return "tutor.current_fact.v2";
    case "problem_solving": return "tutor.solve_problem.v2";
    default: return "tutor.explain_concept.v2";
  }
}

export function resolveTutorDepth(request: TutorRequest): "BRIEF" | "NORMAL" | "DEEP" {
  if (request.depth) return request.depth;
  if (/\b(?:just (?:give|tell)|brief|short answer|answer only)\b/i.test(request.message)) return "BRIEF";
  if (/\b(?:deep|detailed|step[- ]by[- ]step|show all working)\b/i.test(request.message)) return "DEEP";
  return "NORMAL";
}
