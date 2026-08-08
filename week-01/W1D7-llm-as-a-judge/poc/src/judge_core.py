"""
judge_core.py — Core LLM-as-a-Judge logic.

This module is importable independently of main.py so that unit tests
can exercise the rubric, prompt-building, and verdict-parsing logic
without triggering a full end-to-end run.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Data models (pure Python dataclasses — no Pydantic dependency at core level)
# ---------------------------------------------------------------------------

@dataclass
class CriterionVerdict:
    score: int           # 1 = fail, 2 = needs improvement, 3 = pass
    rationale: str = ""  # Required when score < 3

    def __post_init__(self) -> None:
        if self.score not in (1, 2, 3):
            raise ValueError(f"Score must be 1, 2, or 3 — got {self.score}")
        if self.score < 3 and not self.rationale:
            raise ValueError("Rationale is required when score < 3")


@dataclass
class JudgeVerdict:
    criteria: dict[str, CriterionVerdict]
    overall: Literal["pass", "review", "fail"]
    confidence: Literal["high", "medium", "low"]
    rubric_version: str = "v1.0"
    parse_attempts: int = 1

    def passed(self) -> bool:
        return self.overall == "pass"

    def needs_human_review(self) -> bool:
        # Route to human queue if fail, review, low confidence, or any criterion scored 1
        if self.overall in ("fail", "review"):
            return True
        if self.confidence == "low":
            return True
        return any(v.score == 1 for v in self.criteria.values())

    def summary(self) -> str:
        lines = [f"Overall: {self.overall.upper()} (confidence: {self.confidence})"]
        for name, verdict in self.criteria.items():
            flag = "" if verdict.score == 3 else f" | {verdict.rationale}"
            lines.append(f"  {name}: {verdict.score}/3{flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in rubrics — versioned and immutable once deployed
# ---------------------------------------------------------------------------

RUBRICS: dict[str, dict[str, str]] = {
    "v1.0": {
        "relevance": (
            "Does the response directly address the user's question? "
            "Score 1: response addresses a different question entirely. "
            "Score 2: partially addresses the question but misses key aspects. "
            "Score 3: fully addresses all parts of the question."
        ),
        "accuracy": (
            "Are all factual claims in the response correct and consistent with "
            "the reference material (if provided)? "
            "Score 1: contains at least one factually incorrect claim. "
            "Score 2: all claims are correct but omits important context. "
            "Score 3: fully accurate and appropriately contextualised."
        ),
        "completeness": (
            "Does the response include all information necessary for the user "
            "to act on it? "
            "Score 1: missing critical information required to act. "
            "Score 2: adequate but notable omissions present. "
            "Score 3: comprehensive and actionable."
        ),
    }
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, calibrated evaluator for AI-generated responses. "
    "Evaluate the candidate response against each rubric criterion. "
    "Be rigorous — a score of 3 means fully correct, not merely acceptable. "
    "Return ONLY valid JSON matching the specified schema. No prose before or after the JSON."
)

OUTPUT_SCHEMA_HINT = """{
  "criteria": {
    "<criterion_name>": {
      "score": <1|2|3>,
      "rationale": "<required when score < 3, empty string when score = 3>"
    }
  },
  "overall": "<pass|review|fail>",
  "confidence": "<high|medium|low>"
}

Derive "overall":
  - "fail"   if any criterion score = 1
  - "review" if any criterion score = 2 (and none = 1)
  - "pass"   if all criterion scores = 3"""


def build_judge_prompt(
    user_prompt: str,
    candidate_response: str,
    rubric_version: str = "v1.0",
    reference: str = "",
) -> list[dict]:
    """
    Construct the judge prompt message list.

    Wraps the candidate response in sentinel delimiters to mitigate
    prompt injection — content between the sentinels is treated as
    inert data by the judge, not as instructions.
    """
    rubric = RUBRICS.get(rubric_version, RUBRICS["v1.0"])
    rubric_text = "\n".join(
        f"Criterion '{name}': {desc}" for name, desc in rubric.items()
    )

    user_content = f"""RUBRIC (version: {rubric_version}):
{rubric_text}

ORIGINAL REQUEST:
<<<BEGIN_REQUEST>>>
{user_prompt}
<<<END_REQUEST>>>

CANDIDATE RESPONSE:
<<<BEGIN_RESPONSE>>>
{candidate_response}
<<<END_RESPONSE>>>
"""
    if reference:
        user_content += f"""
REFERENCE MATERIAL (ground truth / context):
<<<BEGIN_REFERENCE>>>
{reference}
<<<END_REFERENCE>>>
"""

    user_content += f"\nReturn ONLY this JSON schema:\n{OUTPUT_SCHEMA_HINT}"

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_verdict(
    raw: str,
    rubric_version: str = "v1.0",
    parse_attempts: int = 1,
) -> JudgeVerdict:
    """
    Parse raw judge completion text into a JudgeVerdict.

    Attempts strict JSON parse first. Falls back to extracting the first
    JSON object found in the text (handles models that prepend/append prose).
    Raises ValueError if the schema is not satisfied.
    """
    data: dict | None = None

    # Strict parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: find first {...} block in the output
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        raise ValueError(f"Could not parse JSON from judge output: {raw[:200]}")

    # Validate required top-level keys
    for key in ("criteria", "overall", "confidence"):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in judge output")

    if data["overall"] not in ("pass", "review", "fail"):
        raise ValueError(f"Invalid overall value: {data['overall']}")
    if data["confidence"] not in ("high", "medium", "low"):
        raise ValueError(f"Invalid confidence value: {data['confidence']}")

    criteria: dict[str, CriterionVerdict] = {}
    for name, cv in data["criteria"].items():
        criteria[name] = CriterionVerdict(
            score=int(cv["score"]),
            rationale=cv.get("rationale", ""),
        )

    return JudgeVerdict(
        criteria=criteria,
        overall=data["overall"],
        confidence=data["confidence"],
        rubric_version=rubric_version,
        parse_attempts=parse_attempts,
    )
