"""Prompt templates and the answer cue."""

from __future__ import annotations

from typing import Any, Dict, List

MATH_TEMPLATE = "Solve the following problem step by step. End your response with the final answer in \\boxed{{}}.\n\n{question}"
MC_TEMPLATE = (
    "Answer the following multiple-choice question. Reason step by step, then "
    "end your response with the letter of the correct option in \\boxed{{}} "
    "(for example \\boxed{{A}}).\n\n{question}"
)
QA_TEMPLATE = (
    "Answer the following question. Think step by step, then end your response "
    "with the short final answer in \\boxed{{}}.\n\n{question}"
)
TEMPLATES = {"math": MATH_TEMPLATE, "mc": MC_TEMPLATE, "qa": QA_TEMPLATE}

# Appended after a reasoning prefix and the end-of-reasoning marker to elicit an intermediate answer.
ANSWER_CUE = "The final answer is \\boxed{"


def messages_for(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Single-turn chat for a dataset row, templated by answer type."""
    tmpl = TEMPLATES.get(row.get("type", "math"), MATH_TEMPLATE)
    return [{"role": "user", "content": tmpl.format(question=row["question"])}]
