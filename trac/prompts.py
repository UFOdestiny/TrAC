"""Prompt templates, answer cues, and the follow-up questions of the self-assessment baselines."""
from __future__ import annotations

from typing import Any, Dict, List

MATH_TEMPLATE = (
    "Solve the following problem step by step. "
    "End your response with the final answer in \\boxed{{}}.\n\n{question}"
)
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

# Answer cues appended after the reasoning prefix; cue 0 is the PCE cue, cues 1-2 are the
# paraphrases of the cue-robustness check.
CUES = ["The final answer is \\boxed{",
        "Therefore, the answer is \\boxed{",
        "So the final result is \\boxed{"]

PTRUE_QUESTION = "Is the proposed final answer correct? Respond with a single word: True or False."
VERBAL_CONFIDENCE_QUESTION = ("How confident are you that the proposed final answer is correct? "
                              "Respond only with 'Confidence: X%' where X is an integer from 0 to 100.")


def messages_for(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Single-turn chat for a dataset row, templated by answer type."""
    tmpl = TEMPLATES.get(row.get("type", "math"), MATH_TEMPLATE)
    return [{"role": "user", "content": tmpl.format(question=row["question"])}]


def proposed_answer_messages(row: Dict[str, Any], answer: str, question: str) -> List[Dict[str, str]]:
    """Chat that shows the returned answer and asks a self-assessment question
    (P(True), verbalized confidence)."""
    return messages_for(row) + [
        {"role": "assistant", "content": f"The final answer is \\boxed{{{answer}}}."},
        {"role": "user", "content": question}]
