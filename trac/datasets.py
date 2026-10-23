"""Benchmarks with verifiable answers, read from flat local mirrors under DATASETS_ROOT.

  tag            source (folder under DATASETS_ROOT)                 n     type
  gsm8k          openai/gsm8k (main, test)                           1319  math
  math500        HuggingFaceH4/MATH-500                              500   math
  minerva        math-ai/minervamath                                 272   math
  olympiad       math-ai/olympiadbench (text-only, single answer)    581   math
  aime3          math-ai/aime24 + math-ai/aime25 + MathArena/aime_2026  90  math
  gpqa_diamond   Idavidrein/gpqa (diamond CSV, seeded option shuffle) 198  mc
  mmlupro        TIGER-Lab/MMLU-Pro (category-stratified, seed 2027) 1000 mc
  triviaqa       mandarjoshi/trivia_qa (rc.nocontext validation)     1000 qa

`scripts/download_data.py` creates the mirrors (folder `org_name`). Each row is
{id, question, gold, gold_raw?, aliases?, type, n_choices?, category?}; multiple-choice
questions embed their lettered options.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
from typing import Any, Callable, Dict, List

from trac.answers import normalize_num, normalize_qa
from trac.config import DATASETS_ROOT

Row = Dict[str, Any]


def _dir(repo: str) -> str:
    """Local mirror of a HuggingFace dataset repo `org/name`."""
    p = os.path.join(str(DATASETS_ROOT), repo.replace("/", "_", 1))
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{p} (run scripts/download_data.py)")
    return p


def _fmt_options(opts: List[str]) -> str:
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))


def _scalar(a):
    import numpy as np
    if isinstance(a, (list, tuple, np.ndarray)):
        return a[0] if len(a) else None
    return a


def _math_row(rid: str, question: str, answer: str, **extra) -> Row:
    a = str(answer).strip()
    return {"id": rid, "question": question, "gold": normalize_num(a) or a, "gold_raw": a, "type": "math", **extra}


def load_gsm8k() -> List[Row]:
    import pandas as pd
    p = glob.glob(os.path.join(_dir("openai/gsm8k"), "main", "test-*.parquet"))[0]
    rows = []
    for i, r in pd.read_parquet(p).iterrows():
        m = re.findall(r"####\s*(-?[\d,\.]+)", r["answer"])
        rows.append({"id": f"gsm8k-test-{i}", "question": r["question"],
                     "gold": normalize_num(m[-1] if m else r["answer"]), "type": "math"})
    return rows


def load_math500() -> List[Row]:
    rows = []
    for i, line in enumerate(open(os.path.join(_dir("HuggingFaceH4/MATH-500"), "test.jsonl"))):
        r = json.loads(line)
        rows.append({"id": f"math500-{i}", "question": r["problem"],
                     "gold": normalize_num(r["answer"]) or r["answer"].strip(),
                     "gold_raw": r["answer"].strip(), "type": "math", "level": r.get("level")})
    return rows


def load_minerva() -> List[Row]:
    rows = []
    for i, line in enumerate(open(os.path.join(_dir("math-ai/minervamath"), "test.jsonl"))):
        r = json.loads(line)
        rows.append(_math_row(f"minerva-{i}", r["question"], str(r.get("answer"))))
    return rows


def load_olympiad() -> List[Row]:
    """Text-only, single-answer subset (581 questions)."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(_dir("math-ai/olympiadbench"), "test.parquet"))
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if r.get("image_1") is not None or bool(r.get("is_multiple_answer")):
            continue
        a = _scalar(r.get("final_answer"))
        if a is None:
            continue
        rows.append(_math_row(f"olympiad-{i}", r["question"], str(a)))
    return rows


def load_aime3() -> List[Row]:
    """AIME 2024 + 2025 + 2026 (90 questions). AIME 2026 postdates every backbone's
    release, so it doubles as a contamination-free slice."""
    import pandas as pd
    rows = []
    p = glob.glob(os.path.join(_dir("math-ai/aime24"), "test-*.parquet"))[0]
    for _, r in pd.read_parquet(p).iterrows():          # gold inside the boxed solution
        m = re.search(r"\\boxed\{([^}]*)\}", str(r["solution"]))
        rows.append(_math_row(f"aime24-{r['id']}", r["problem"], m.group(1) if m else str(r["solution"])))
    for line in open(os.path.join(_dir("math-ai/aime25"), "test.jsonl")):
        r = json.loads(line)
        rows.append(_math_row(f"aime25-{r['id']}", r["problem"], str(r["answer"])))
    p = os.path.join(_dir("MathArena/aime_2026"), "data", "train-00000-of-00001.parquet")
    for _, r in pd.read_parquet(p).iterrows():
        rows.append(_math_row(f"aime26-{r['problem_idx']}", r["problem"], str(r["answer"])))
    return rows


def load_gpqa_diamond() -> List[Row]:
    """Official GPQA-Diamond CSV; options shuffled once with a per-question seed so the
    gold letter is uniform."""
    import pandas as pd
    df = pd.read_csv(os.path.join(_dir("Idavidrein/gpqa"), "gpqa_diamond.csv"))
    rows = []
    for i, r in df.iterrows():
        opts = [str(r["Correct Answer"]).strip()] + [str(r[f"Incorrect Answer {k}"]).strip() for k in (1, 2, 3)]
        order = list(range(4))
        random.Random(1000 + i).shuffle(order)
        q = str(r["Question"]).strip() + "\n\n" + _fmt_options([opts[j] for j in order])
        rows.append({"id": f"gpqa-{i}", "question": q, "gold": chr(65 + order.index(0)), "type": "mc", "n_choices": 4})
    return rows


def load_mmlupro(n: int = 1000) -> List[Row]:
    """MMLU-Pro test, category-stratified seeded sample of n questions (up to 10 options)."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(_dir("TIGER-Lab/MMLU-Pro"), "data", "test-00000-of-00001.parquet"))
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=2027)
             for _, g in df.groupby("category", sort=True)]
    df = pd.concat(parts).sort_values("question_id").head(n)
    rows = []
    for _, r in df.iterrows():
        opts = [str(o) for o in r["options"]]
        rows.append({"id": f"mmlupro-{r['question_id']}", "question": str(r["question"]).strip() + "\n\n" + _fmt_options(opts),
                     "gold": str(r["answer"]).strip().upper(), "type": "mc", "n_choices": len(opts),
                     "category": r["category"]})
    return rows


def load_triviaqa(n: int = 1000) -> List[Row]:
    """TriviaQA rc.nocontext validation, seeded sample of n questions; verified by
    normalized exact match against the alias list."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(_dir("mandarjoshi/trivia_qa"), "rc.nocontext", "validation-00000-of-00001.parquet"))
    df = df.sample(n=n, random_state=2027).sort_values("question_id")
    rows = []
    for _, r in df.iterrows():
        ans = r["answer"]
        aliases = sorted({normalize_qa(a) for a in list(ans["aliases"]) + list(ans["normalized_aliases"])
                          + [ans["value"]] if str(a).strip()})
        rows.append({"id": f"triviaqa-{r['question_id']}", "question": str(r["question"]).strip(),
                     "gold": str(ans["value"]), "aliases": aliases, "type": "qa"})
    return rows


LOADERS: Dict[str, Callable[[], List[Row]]] = {
    "gsm8k": load_gsm8k, "math500": load_math500, "minerva": load_minerva, "olympiad": load_olympiad,
    "aime3": load_aime3, "gpqa_diamond": load_gpqa_diamond, "mmlupro": load_mmlupro, "triviaqa": load_triviaqa,
}
MATH = ("gsm8k", "math500", "minerva", "olympiad", "aime3")


def load_dataset(name: str, limit: int = 0) -> List[Row]:
    """Rows of a benchmark tag, or `mix:ds1,ds2,...:N` (N seeded-random questions from each,
    used for smoke tests)."""
    name = name.lower()
    if name.startswith("mix:"):
        _, names, n = name.split(":")
        rows = []
        for d in names.split(","):
            part = load_dataset(d)
            rows += random.Random(2027).sample(part, min(int(n), len(part)))
    else:
        rows = LOADERS[name]()
    return rows[:limit] if limit else rows
