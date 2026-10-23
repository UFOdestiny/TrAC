import pytest

from trac import answers as A

MATH = {"type": "math", "gold": "12", "gold_raw": "12"}
MC = {"type": "mc", "gold": "C", "n_choices": 4}
QA = {"type": "qa", "gold": "Paris", "aliases": ["paris", "city of paris"]}


def test_last_boxed_is_brace_balanced():
    assert A.last_boxed(r"x \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert A.last_boxed("no answer") is None


def test_boxed_continuation():
    assert A.boxed_continuation(r"\frac{3}{4}} and more") == r"\frac{3}{4}"
    assert A.boxed_continuation("42") is None


@pytest.mark.parametrize("raw,canon", [("1,234", "1234"), ("(2,4)", "(2,4)"), ("$5.0$", "5"), (r"3\,\text{cm}.", r"3 \text{cm}")])
def test_canon_math(raw, canon):
    assert A.canon_math(raw) == canon


def test_math_equivalence():
    assert A.equiv(MATH, "0.5", r"\frac{1}{2}")
    assert A.equiv(MATH, "x(t)=2t", "x_p(t)=2t")
    assert not A.equiv(MATH, "3", "4")
    assert not A.equiv(MATH, None, "4")


def test_extract_answer_by_type():
    assert A.extract_answer(MATH, r"so \boxed{1,000}") == "1000"
    assert A.extract_answer(MC, r"The answer is \boxed{(c)}") == "C"
    assert A.extract_answer(QA, r"\boxed{Paris}") == "Paris"
    assert A.extract_answer(MATH, "") is None


def test_extract_head():
    assert A.extract_head(MATH, r"\frac{1}{2}}. Done") == r"\frac{1}{2}"
    assert A.extract_head(MC, "B} because") == "B"
    assert A.extract_head(MC, r"\text{(D)}}") == "D"
    assert A.extract_head(QA, "Paris} is") == "Paris"
    assert A.extract_head(MATH, "}") is None


def test_verification():
    assert A.verify_answer_text(MATH, r"\boxed{12}") == 1
    assert A.verify_answer_text(MATH, r"\boxed{13}") == 0
    assert A.verify_answer_text(MC, r"\boxed{C}") == 1
    assert A.verify_answer_text(QA, r"\boxed{The City of Paris}") == 1
    assert A.verify_answer_text(QA, "") == 0


def test_scientific_gold():
    assert A.gold_for_verify("5.89e-7") == r"5.89\times10^{-7}"


def test_normalize_qa():
    assert A.normalize_qa("The  Beatles (band)!") == "beatles"


def test_surface_answer():
    row = {"answer": "1/2", "type": "math", "final": r"\boxed{\frac{1}{2}}"}
    assert A.surface_answer(row) == r"\frac{1}{2}"
    assert A.surface_answer({**row, "answer": None}) is None
