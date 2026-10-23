"""Backbone registry and token-level reasoning-format helpers.

Every probe is built at the TOKEN level from the stored original generation:
    prompt_ids + output_ids[:reason_end] + tok(close + cue)
so the completed reasoning prefix is byte-identical to what the model produced
(no re-segmentation, no left truncation) and the vLLM prefix cache is reused.

Formats
  think    Qwen3 / Olmo-3-Think / R1-Distill: reasoning inside <think> ... </think>
  harmony  gpt-oss: reasoning in the `analysis` channel, answer in the `final` channel
  plain    instruct models (Llama-3.1, Ministral): chain of thought in the response; the
           reasoning body is the response up to the line holding the first \\boxed{}

Prompts are built as token ids from the chat template (tokenize=True): re-encoding a
rendered template string is not safe for every tokenizer (Ministral's tekken backend
encodes [INST] as text).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from trac.config import model_path


@dataclass(frozen=True)
class Backbone:
    key: str                 # short key embedded in cache tags, e.g. math500_qwen8b_k8
    name: str                # folder under MODELS_ROOT
    family: str
    fmt: str                 # "think" | "harmony" | "plain"
    end_marker: str          # string that terminates the reasoning body in the raw output
    reason_open: str         # string that opens an (empty) reasoning block
    close: str               # appended after the reasoning body, before the answer cue
    max_model_len: int
    template_kwargs: Tuple[Tuple[str, object], ...] = ()
    max_new: int = 32768                       # generation budget (new tokens)
    llm_kwargs: Tuple[Tuple[str, object], ...] = ()


# The paper's matrix uses qwen8b, llama8b, and ministral8b; the others are registered and
# smoke-tested only.
BACKBONES = {
    "qwen8b": Backbone("qwen8b", "Qwen3-8B", "Qwen", "think",
                       "</think>", "<think>\n", "</think>\n\n", 40960,
                       (("enable_thinking", True),)),
    "llama8b": Backbone("llama8b", "Llama-3.1-8B-Instruct", "Llama", "plain",
                        "\\boxed{", "", "", 32768, max_new=16384),
    "ministral8b": Backbone("ministral8b", "Ministral-8B-Instruct-2410", "Mistral", "plain",
                            "\\boxed{", "", "", 32768, max_new=16384,
                            llm_kwargs=(("tokenizer_mode", "mistral"), ("config_format", "mistral"),
                                        ("load_format", "mistral"))),
    "olmo7b": Backbone("olmo7b", "Olmo-3-7B-Think", "OLMo", "think",
                       "</think>", "<think>", "</think>\n\n", 40960),
    "gptoss20b": Backbone("gptoss20b", "gpt-oss-20b", "gpt-oss", "harmony",
                          "<|end|>", "<|channel|>analysis<|message|>",
                          "<|end|><|start|>assistant<|channel|>final<|message|>", 40960,
                          (("reasoning_effort", "medium"),)),
    "r1qwen7b": Backbone("r1qwen7b", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek", "think",
                         "</think>", "<think>\n", "</think>\n\n", 40960),
}

_MATH_OPENERS = {"", "\\[", "$$", "\\(", "\\begin{align*}", "\\begin{aligned}", "\\begin{equation*}"}

# special tokens that may trail the answer section in a raw (non-skipped) decode
_TRAILERS = ["<|im_end|>", "<|endoftext|>", "<|return|>", "<|end|>",
             "<｜end▁of▁sentence｜>", "</s>"]


def get(key: str) -> Backbone:
    return BACKBONES[key]


def from_tag(tag: str) -> Optional[Backbone]:
    """Backbone of a cache tag (the key is one of its underscore-separated tokens)."""
    return next((b for b in BACKBONES.values() if b.key in tag.split("_")), None)


def resolve(key: Optional[str], tag: str) -> Backbone:
    """Backbone named by `key`, else the one embedded in `tag`."""
    bb = get(key) if key else from_tag(tag)
    if bb is None:
        raise ValueError(f"cannot infer the backbone of {tag!r}; pass --model")
    return bb


def load_tokenizer(bb: Backbone):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path(bb.name), trust_remote_code=True)


def make_llm(bb: Backbone, gpu_mem: float, seed: int = 1234, **kw):
    """vLLM engine for a backbone (model-specific loading options included)."""
    from vllm import LLM
    args = dict(model=model_path(bb.name), trust_remote_code=True, dtype="bfloat16",
                gpu_memory_utilization=gpu_mem, max_model_len=bb.max_model_len, seed=seed,
                enable_prefix_caching=True)
    args.update(dict(bb.llm_kwargs)); args.update(kw)
    return LLM(**args)


def prompt_ids(tok, bb: Backbone, messages) -> List[int]:
    """Chat-template token ids ending at the assistant generation prompt (reasoning on)."""
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  **dict(bb.template_kwargs))
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def encode(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


def direct_ids(tok, bb: Backbone, messages) -> List[int]:
    """Prompt with an EMPTY reasoning block, ready for an answer cue (question-only
    probe, P(True), verbalized confidence). Opens the block only if the template
    has not already."""
    ids = prompt_ids(tok, bb, messages)
    if bb.fmt == "plain":
        return ids
    opened = tok.decode(ids[-8:], skip_special_tokens=False).rstrip().endswith(bb.reason_open.strip())
    return ids + encode(tok, ("" if opened else bb.reason_open) + bb.close)


def reasoning_end_char(bb: Backbone, raw: str) -> Optional[int]:
    """Char offset where the reasoning body ends in the raw output, or None if the
    reasoning never closed (truncated generation / no boxed answer for plain CoT)."""
    i = raw.find(bb.end_marker)
    if i < 0:
        return None
    if bb.fmt == "plain":                     # cut before the line that states the answer,
        c = raw.rfind("\n", 0, i) + 1         # and before any display-math opener lines
        while c > 0:
            prev = raw.rfind("\n", 0, c - 1) + 1
            if raw[prev:c].strip() in _MATH_OPENERS:
                c = prev
            else:
                break
        return c
    return i


def reasoning_body(bb: Backbone, raw: str) -> str:
    """Reasoning text of a raw output (all of it when the reasoning never closed)."""
    c = reasoning_end_char(bb, raw)
    return raw[:c] if c is not None else raw


def final_section(bb: Backbone, raw: str) -> str:
    """The returned answer section (post-reasoning), special tokens stripped."""
    if bb.fmt == "harmony":
        tag = "<|channel|>final<|message|>"
        i = raw.rfind(tag)
        if i < 0:
            return ""
        out = raw[i + len(tag):]
    elif bb.fmt == "plain":
        c = reasoning_end_char(bb, raw)
        out = raw[c:] if c is not None else raw  # no boxed answer: let extraction fall back
    else:
        i = raw.find(bb.end_marker)
        if i < 0:
            return ""
        out = raw[i + len(bb.end_marker):]
    for t in _TRAILERS:
        out = out.split(t)[0]
    return out.strip()


def final_for(bb: Backbone, raw: str, finish) -> str:
    """Returned answer section; empty for a generation cut off before it answered."""
    closed = reasoning_end_char(bb, raw) is not None
    return final_section(bb, raw) if (closed or finish != "length") else ""


def token_cut(tok, ids: List[int], char_end: int) -> int:
    """Largest k such that decode(ids[:k]) has length <= char_end (binary search).
    With a single-token end marker this is exactly the marker position."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=False)) <= char_end:
            lo = mid
        else:
            hi = mid - 1
    return lo


def reasoning_prefix(tok, bb: Backbone, out_ids: List[int], raw: str) -> Tuple[List[int], bool]:
    """Output token ids of the completed reasoning body (+ any sub-token remainder
    re-encoded) and whether the reasoning closed on its own."""
    c = reasoning_end_char(bb, raw)
    if c is None:
        return list(out_ids), False
    k = token_cut(tok, out_ids, c)
    rest = raw[len(tok.decode(out_ids[:k], skip_special_tokens=False)):c]
    return list(out_ids[:k]) + (encode(tok, rest) if rest else []), True
