from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException


class CommandWordDrillService:
    def __init__(self, db, model) -> None:
        self._db = db
        self._model = model

    def build_drill(
        self,
        *,
        objective_id: str,
        requested_words: list[str] | None = None,
    ) -> dict[str, Any]:
        objective = self._fetch_objective(objective_id)
        command_words = [
            item.lower().strip()
            for item in (requested_words or objective.get("command_words") or [])
            if str(item).strip()
        ]
        if not command_words:
            command_words = ["state", "describe", "explain"]
        selected_words = command_words[:3]

        title = str(objective.get("title") or objective.get("topic") or objective_id)
        description = str(objective.get("description") or "").strip()
        prompt_seed = title if not description else f"{title}: {description}"

        cards = []
        for word in selected_words:
            cards.append(
                {
                    "command_word": word,
                    "prompt": f"{word.upper()} {prompt_seed}.",
                    "depth_expectation": self._depth_expectation(word),
                }
            )

        return {
            "objective_id": objective_id,
            "board": objective.get("board", ""),
            "subject": objective.get("subject", ""),
            "paper": objective.get("paper", ""),
            "topic": objective.get("topic", ""),
            "sub_topic": objective.get("sub_topic", ""),
            "title": title,
            "description": description,
            "cards": cards,
            "source_type": "COMMAND_WORD_DRILL",
        }

    async def evaluate_drill(
        self,
        *,
        objective_id: str,
        prompt_cards: list[dict[str, Any]],
        responses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._model is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini is required for command-word drill grading",
            )

        objective = self._fetch_objective(objective_id)
        heuristic_items = [
            self._score_depth(
                command_word=str(item.get("command_word", "")),
                response=next(
                    (
                        str(candidate.get("response", ""))
                        for candidate in responses
                        if str(candidate.get("command_word", "")).lower().strip()
                        == str(item.get("command_word", "")).lower().strip()
                    ),
                    "",
                ),
            )
            for item in prompt_cards
        ]
        prompt = f"""
Role: Senior examiner for command-word drilling.
Objective: {objective.get("code", objective_id)} | {objective.get("title", "")}
Description: {objective.get("description", "")}
Prompt Cards: {json.dumps(prompt_cards, ensure_ascii=True)}
Student Responses: {json.dumps(responses, ensure_ascii=True)}
Heuristic Depth Analysis: {json.dumps(heuristic_items, ensure_ascii=True)}

Depth Rules:
- state: concise, exact fact or definition only, ideally 1 to 5 words
- describe: accurate features or sequence without causal reasoning
- explain: causal chain or why/how logic using because/therefore/so/causes/results in
- contrast: explicit differences between two valid elements
- suggest: plausible inference grounded in the scenario

Return strict JSON with keys:
overall_score, feedback, items

Each item must include:
command_word, awarded_score, max_score, depth_satisfied, missing_depth, feedback
"""

        try:
            response = await self._model.generate_content_async(prompt)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Gemini drill evaluation failed: {exc}"
            ) from exc

        raw_text = getattr(response, "text", "").strip()
        if not raw_text:
            raise HTTPException(
                status_code=502, detail="Gemini returned an empty drill evaluation"
            )

        candidate = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502, detail="Drill evaluation response was not valid JSON"
            ) from exc

        parsed.setdefault("overall_score", 0)
        parsed.setdefault("feedback", "")
        parsed_items = list(parsed.get("items", []))
        if not parsed_items:
            parsed_items = []
        merged_items = []
        for heuristic in heuristic_items:
            existing = next(
                (
                    item
                    for item in parsed_items
                    if str(item.get("command_word", "")).lower().strip()
                    == str(heuristic.get("command_word", "")).lower().strip()
                ),
                {},
            )
            merged = {
                "command_word": heuristic["command_word"],
                "awarded_score": int(
                    existing.get("awarded_score", heuristic["awarded_score"])
                ),
                "max_score": int(existing.get("max_score", heuristic["max_score"])),
                "depth_satisfied": bool(
                    existing.get("depth_satisfied", heuristic["depth_satisfied"])
                ),
                "missing_depth": list(
                    existing.get("missing_depth", heuristic["missing_depth"])
                ),
                "feedback": str(existing.get("feedback", heuristic["feedback"])),
                "depth_score": float(
                    existing.get("depth_score", heuristic["depth_score"])
                ),
            }
            merged_items.append(merged)
        parsed["items"] = merged_items
        if merged_items and not parsed.get("overall_score"):
            parsed["overall_score"] = round(
                sum(float(item["awarded_score"]) for item in merged_items)
                / max(sum(float(item["max_score"]) for item in merged_items), 1.0),
                4,
            )
        return parsed

    def _fetch_objective(self, objective_id: str) -> dict[str, Any]:
        snapshots = (
            self._db.collection("syllabus_maps")
            .where("objective_id", "==", objective_id)
            .limit(1)
            .stream()
        )
        for snapshot in snapshots:
            return snapshot.to_dict() or {}

        snapshots = (
            self._db.collection("syllabus_maps")
            .where("code", "==", objective_id)
            .limit(1)
            .stream()
        )
        for snapshot in snapshots:
            return snapshot.to_dict() or {}

        raise HTTPException(status_code=404, detail="Objective not found")

    def _depth_expectation(self, command_word: str) -> str:
        return {
            "state": "A precise fact or definition in roughly 1 to 5 words.",
            "describe": "Observable features or sequence, but not a why-chain.",
            "explain": "Cause-and-effect reasoning or a linked why/how chain.",
            "contrast": "A direct comparison highlighting meaningful differences.",
            "suggest": "A plausible inference grounded in the context and objective.",
        }.get(
            command_word.lower(), "Respond with the depth expected by the command word."
        )

    def _score_depth(self, *, command_word: str, response: str) -> dict[str, Any]:
        normalized_word = re.sub(r"[^a-z]", "", command_word.lower())
        normalized_response = " ".join(response.strip().split())
        words = re.findall(r"[A-Za-z0-9']+", normalized_response.lower())
        word_count = len(words)
        lower_response = normalized_response.lower()
        sequence_markers = ["first", "then", "next", "after", "before", "finally"]
        causal_markers = [
            "because",
            "therefore",
            "thus",
            "hence",
            "since",
            "due to",
            "as a result",
            "results in",
            "leads to",
            "causes",
        ]
        missing_depth: list[str] = []
        depth_score = 0.0
        satisfied = False

        if normalized_word == "state":
            satisfied = 1 <= word_count <= 5
            if word_count == 0:
                missing_depth.append("No fact stated.")
            elif word_count > 5:
                missing_depth.append("Make the answer shorter and more exact.")
            depth_score = 1.0 if satisfied else (0.5 if word_count else 0.0)
        elif normalized_word == "describe":
            has_sequence = any(marker in lower_response for marker in sequence_markers)
            satisfied = word_count >= 6 and has_sequence
            if word_count < 6:
                missing_depth.append("Add more observable detail.")
            if not has_sequence:
                missing_depth.append("Show the sequence of events.")
            depth_score = min(
                1.0,
                (0.45 if word_count >= 6 else word_count / 12.0)
                + (0.55 if has_sequence else 0.0),
            )
        elif normalized_word == "explain":
            has_causal = any(marker in lower_response for marker in causal_markers)
            satisfied = word_count >= 8 and has_causal
            if word_count < 8:
                missing_depth.append("Add a fuller causal chain.")
            if not has_causal:
                missing_depth.append(
                    "Use because, therefore, or equivalent cause-effect language."
                )
            depth_score = min(
                1.0,
                (0.35 if word_count >= 8 else word_count / 16.0)
                + (0.65 if has_causal else 0.0),
            )
        else:
            satisfied = word_count > 0
            depth_score = 1.0 if satisfied else 0.0
            if not satisfied:
                missing_depth.append("No answer supplied.")

        awarded_score = 2 if satisfied else (1 if depth_score >= 0.45 else 0)
        feedback = (
            "Depth is correct for the command word."
            if satisfied
            else " ".join(missing_depth) or "Depth does not match the command word."
        )
        return {
            "command_word": command_word,
            "awarded_score": awarded_score,
            "max_score": 2,
            "depth_satisfied": satisfied,
            "missing_depth": missing_depth,
            "feedback": feedback,
            "depth_score": round(depth_score, 4),
        }
