from trac import backbones as M

QWEN, LLAMA = M.get("qwen8b"), M.get("llama8b")


def test_from_tag():
    assert M.from_tag("math500_qwen8b_k8").key == "qwen8b"
    assert M.from_tag("dm_g_minerva_ministral8b").key == "ministral8b"
    assert M.from_tag("nothing_here") is None


def test_think_format():
    raw = "<think>\nwork</think>\n\nThe answer is \\boxed{3}.<|im_end|>"
    assert raw[:M.reasoning_end_char(QWEN, raw)] == "<think>\nwork"
    assert M.final_section(QWEN, raw) == "The answer is \\boxed{3}."
    assert M.final_for(QWEN, "<think>\nunfinished", "length") == ""


def test_plain_format_cuts_before_answer_line():
    raw = "Step 1.\nStep 2.\n$$\n\\boxed{3}\n$$"
    c = M.reasoning_end_char(LLAMA, raw)
    assert raw[:c] == "Step 1.\nStep 2.\n"
    assert M.reasoning_body(LLAMA, "no answer") == "no answer"
