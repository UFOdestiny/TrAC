"""Answer extraction, typed equivalence, and deterministic verification.

Three answer types share one interface (rows carry `type`):
  math  boxed LaTeX answers, compared with math_verify (numbers compared directly)
  mc    option letters, exact match
  qa    short open answers, SQuAD/TriviaQA normalization against the alias list

The same equivalence is used for correctness labels, re-elicitation agreement, and
sample votes; labels never enter the reference-free comparisons.
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Any, Dict, Optional

_ABC = string.ascii_uppercase


# ----------------------------------------------------------------------------- boxed text

def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} in `text`, brace-balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1: j]
    return None


def boxed_continuation(text: str) -> Optional[str]:
    """Answer text of a continuation that starts right after '\\boxed{': everything up to
    the matching closing brace, or None if the brace never closes."""
    depth = 1
    for k, ch in enumerate(text):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return text[:k]
    return None


# ----------------------------------------------------------------------------- math

def normalize_num(s: Optional[str]) -> Optional[str]:
    """Normalize a numeric string for exact comparison."""
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.strip()
    s = s.rstrip(".")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        pass
    return s if s else None


def fallback_math_answer(text: str) -> Optional[str]:
    """Numeric answer without math_verify: \\boxed{}, then 'answer is X', then '#### X',
    then the last number."""
    b = last_boxed(text)
    if b is not None:
        n = normalize_num(b)
        if n is not None:
            return n
    m = re.findall(r"answer\s*(?:is|:)?\s*\$?(-?[\d,\.]+)", text, flags=re.IGNORECASE)
    if m:
        return normalize_num(m[-1])
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return normalize_num(m[-1])
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if m:
        return normalize_num(m[-1])
    return None


def numeric_match(pred: Optional[str], gold: Optional[str]) -> int:
    if pred is None or gold is None:
        return 0
    if pred == gold:
        return 1
    try:
        if abs(float(pred) - float(gold)) < 1e-6:
            return 1
    except Exception:
        pass
    return 0


def canon_math(s) -> Optional[str]:
    """Canonical surface form of a math answer, safe for math_verify: removes LaTeX
    spacing macros and $ delimiters, and drops thousands separators only from pure
    numbers (so tuples like (2,4) keep their commas)."""
    if s is None:
        return None
    s = re.sub(r"\\[,;:! ]", " ", str(s)).replace("~", " ").replace("\\$", "").replace("$", "")
    s = s.replace("\\displaystyle", "").strip().rstrip(".").strip()
    s = " ".join(s.split())
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        s = str(int(f)) if f == int(f) and abs(f) < 1e15 else str(f)
    return s or None


def gold_for_verify(gold_raw: str) -> str:
    """Gold string for math_verify: plain scientific notation (5.89e-7) is rewritten
    as LaTeX so it is not parsed as Euler's e."""
    g = str(gold_raw).strip()
    m = re.fullmatch(r"(-?\d*\.?\d+)\s*[eE]\s*([+-]?\d+)", g)
    return f"{m.group(1)}\\times10^{{{int(m.group(2))}}}" if m else g


_MV = None


def math_verify():
    """(parse, verify) from math_verify, or False when it is not installed."""
    global _MV
    if _MV is None:
        try:
            from math_verify import parse as _p
            from math_verify import verify as _v
            _MV = (_p, _v)
        except Exception:
            _MV = False
    return _MV


_eqcache: Dict[tuple, bool] = {}
_parsecache: Dict[str, Any] = {}


def _parsed(s: str):
    """math_verify parse of "$s$", cached (parsing dominates the cost of math_equal)."""
    if s not in _parsecache:
        _p, _ = math_verify()
        _parsecache[s] = _p("$" + s + "$", parsing_timeout=1)
    return _parsecache[s]


def _num(s: str):
    try:
        return float(s.replace(",", "")) if len(s) < 40 else None
    except ValueError:
        return None


def math_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Symmetric, cached math equivalence of two canonical answers."""
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    key = (a, b) if a <= b else (b, a)
    if key in _eqcache:
        return _eqcache[key]
    fa, fb = _num(a), _num(b)
    if fa is not None and fb is not None:          # both plain numbers: no symbolic parse needed
        val = abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
        _eqcache[key] = val
        return val
    mv = math_verify()
    val = False
    if mv:
        _p, _v = mv
        try:
            val = bool(_v(_parsed(a), _parsed(b), timeout_seconds=1))
        except Exception:
            val = False
    _eqcache[key] = val
    return val


# ----------------------------------------------------------------------------- multiple choice

def extract_choice(text: str, n_choices: int = 26) -> Optional[str]:
    """Option letter of a generation. Priority: \\boxed{X}, 'answer is (X)' / 'answer: X',
    a parenthesised '(X)', else the last standalone capital letter within range."""
    valid = set(_ABC[:n_choices]) if n_choices else set(_ABC)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    m = re.search(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([A-Za-z])\)?\b", text, flags=re.IGNORECASE)
    for c in reversed(m):
        if c.upper() in valid:
            return c.upper()
    m = re.findall(r"\(([A-Za-z])\)", text)
    for c in reversed(m):
        if c.upper() in valid:
            return c.upper()
    m = re.findall(r"\b([A-Z])\b", text)
    for c in reversed(m):
        if c in valid:
            return c
    return None


def verify_choice(row: Dict[str, Any], generation_text: str) -> int:
    pred = extract_choice(generation_text, row.get("n_choices", 26))
    gold = str(row.get("gold", "")).strip().upper()[:1]
    return int(pred is not None and pred == gold)


# ----------------------------------------------------------------------------- open short answers

def normalize_qa(s) -> str:
    """SQuAD/TriviaQA answer normalization: lowercase, drop punctuation and articles,
    collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)          # unicode hyphens/dashes
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\([^)]*\)", " ", s)                       # parenthetical asides, e.g. "(MMORPG)"
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_short_answer(text: str) -> Optional[str]:
    """Short free-form answer: last \\boxed{} content, else an 'answer is' line."""
    b = last_boxed(text)
    if b is not None and b.strip():
        return b.strip()
    m = re.findall(r"answer\s*(?:is|:)\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".")
    return None


def verify_qa(row: Dict[str, Any], generation_text: str) -> int:
    pred = extract_short_answer(generation_text)
    if pred is None:
        return 0
    golds = {normalize_qa(a) for a in (row.get("aliases") or [])} | {normalize_qa(row.get("gold", ""))}
    return int(normalize_qa(pred) in golds - {""})


# ----------------------------------------------------------------------------- typed interface

def extract_answer(row: Dict[str, Any], text: str):
    """Returned answer from a post-reasoning answer section, typed by task."""
    if not text:
        return None
    typ = row.get("type", "math")
    if typ == "mc":
        return extract_choice(text, row.get("n_choices") or 26)
    if typ == "qa":
        return extract_short_answer(text)
    b = last_boxed(text)
    if b is not None and b.strip():
        return canon_math(b)
    return fallback_math_answer(text)


def extract_head(row: Dict[str, Any], head: str):
    """Answer from a probe continuation that starts right after '\\boxed{'."""
    depth, out = 1, []
    for ch in head:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    inner = "".join(out).strip()
    if not inner:
        return None
    typ = row.get("type", "math")
    if typ == "mc":
        flat = re.sub(r"\\text\{([^}]*)\}", r"\1", inner).strip()
        valid = _ABC[:row.get("n_choices") or 26]
        m = re.match(r"^\(?([A-Za-z])\)?(?=[\s.):,]|$)", flat)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
        return extract_choice("\\boxed{" + flat + "}", row.get("n_choices") or 26)
    if typ == "qa":
        return inner
    return canon_math(inner)


def equiv(row: Dict[str, Any], a, b) -> bool:
    """Reference-free answer equivalence (labels never enter this)."""
    if a is None or b is None:
        return False
    typ = row.get("type", "math")
    if typ == "mc":
        return str(a).upper() == str(b).upper()
    if typ == "qa":
        return normalize_qa(a) == normalize_qa(b)
    a, b = str(a), str(b)
    if math_equal(a, b):
        return True
    # "x(t)=expr" vs "x_p(t)=expr": compare right-hand sides when both are equations
    if "=" in a and "=" in b:
        return math_equal(a.rsplit("=", 1)[1].strip(), b.rsplit("=", 1)[1].strip())
    return False


def verify(row: Dict[str, Any], generation_text: str) -> int:
    """Correctness of a math or multiple-choice text: exact letter for `mc`, math_verify
    against the gold with a numeric fallback for `math`."""
    if row.get("type") == "mc":
        return verify_choice(row, generation_text)
    gold_raw = row.get("gold_raw") or row.get("gold")
    mv = math_verify()
    if mv and gold_raw:
        _p, _v = mv
        try:
            gold_parsed = _p("$" + gold_for_verify(gold_raw) + "$")
            pred_parsed = _p(generation_text)
            if _v(gold_parsed, pred_parsed):
                return 1
        except Exception:
            pass
    return numeric_match(fallback_math_answer(generation_text), row.get("gold"))


def verify_answer_text(row: Dict[str, Any], text: str) -> int:
    """Deterministic correctness of a post-reasoning answer section."""
    if not text:
        return 0
    if row.get("type") == "qa":
        return verify_qa(row, text)
    return verify(row, text)


def surface_answer(row: Dict[str, Any]) -> Optional[str]:
    """Surface form of a record's returned answer, as written inside \\boxed{} (used for
    teacher forcing and as the proposed answer of P(True) / verbalized confidence)."""
    if row["answer"] is None:
        return None
    if row["type"] == "mc":
        return str(row["answer"])
    b = last_boxed(row["final"] or "")
    return b.strip() if b and b.strip() else str(row["answer"])
