"""Helpers shared by the vLLM probe stages."""
from __future__ import annotations

from typing import Dict, List, Sequence

from trac import backbones as M
from trac.prompts import CUES


def chosen_logprobs(comp) -> List[float]:
    """Log-probability of every generated token, from vLLM flat logprobs."""
    ids = list(comp.token_ids)
    lp = []
    fl = comp.logprobs
    if fl is not None:
        for p in range(min(len(ids), len(fl.start_indices))):
            for j in range(fl.start_indices[p], fl.end_indices[p]):
                if fl.token_ids[j] == ids[p]:
                    lp.append(fl.logprobs[j]); break
    return lp


def answer_span(tok, ids) -> int:
    """Number of decoded tokens that carry the boxed answer: tokens up to the one
    holding the closing brace, which counts only if it also carries answer text."""
    depth = 1
    for n, t in enumerate(ids):
        piece = tok.decode([t], skip_special_tokens=False)
        for k, ch in enumerate(piece):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                return max(1, n + (1 if piece[:k].strip() else 0))
    return len(ids)


def cue_ids(tok, bb: M.Backbone, cue: int = 0) -> List[int]:
    """Token ids of `close + cue`, appended after a reasoning prefix."""
    return M.encode(tok, bb.close + CUES[cue])


def answer_ids(tok, bb: M.Backbone, answer: str) -> List[int]:
    """Token ids of `answer}` as they follow close + cue 0 (tokenized jointly with the cue
    so the answer gets its in-context tokens)."""
    cc = cue_ids(tok, bb)
    joint = M.encode(tok, bb.close + CUES[0] + answer + "}")
    return joint[len(cc):] if joint[:len(cc)] == cc else M.encode(tok, answer + "}")


class NewlineTokens:
    """Cached test of whether a token id decodes to text containing a line break."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def __call__(self, t: int) -> bool:
        if t not in self.cache:
            self.cache[t] = "\n" in self.tok.decode([t], skip_special_tokens=False)
        return self.cache[t]


def checkpoints(body: Sequence[int], fracs: Sequence[float], has_nl: NewlineTokens) -> List[int]:
    """Increasing checkpoint lengths in (0, T): each fraction of the body, moved back (at
    most 64 tokens) to just after a line break when there is one."""
    T, cks = len(body), []
    for f in fracs:
        k = int(round(f * T))
        if k < 1 or k >= T:
            continue
        j = k
        while j > max(1, k - 64) and not has_nl(body[j - 1]):
            j -= 1
        k = j if has_nl(body[j - 1]) else k
        if 0 < k < T and (not cks or k > cks[-1]):
            cks.append(k)
    return cks


def even_fractions(n: int, decimals: int = 0) -> List[float]:
    """n evenly spaced fractions (j + 1) / (n + 1), optionally rounded."""
    fr = [(j + 1) / (n + 1) for j in range(n)]
    return [round(f, decimals) for f in fr] if decimals else fr


def token_chunks(items: List[Dict], budget: int, key: str = "prefix") -> List[List[Dict]]:
    """Consecutive groups of items whose total `key` length stays within `budget` tokens
    (each group's prefixes fit the KV cache together)."""
    chunks, cur, tot = [], [], 0
    for it in items:
        if cur and tot + len(it[key]) > budget:
            chunks.append(cur); cur, tot = [], 0
        cur.append(it); tot += len(it[key])
    if cur:
        chunks.append(cur)
    return chunks


def score_token_params(t: int):
    """One-token request that returns the log-probability of token `t` at the next position
    (teacher forcing that keeps prefix caching on, unlike prompt_logprobs)."""
    from vllm import SamplingParams
    return SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=1, logprob_token_ids=[t],
                          flat_logprobs=True)


def scored_logprob(out, t: int):
    """Log-probability of token `t` from a `score_token_params` request."""
    fl = out.outputs[0].logprobs
    if fl is None:
        return None
    for j in range(fl.start_indices[0], fl.end_indices[0]):
        if fl.token_ids[j] == t:
            return float(fl.logprobs[j])
    return None
