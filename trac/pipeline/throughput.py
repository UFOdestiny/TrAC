"""Paired cost audit on one GPU: fresh original traces, then on exactly those traces every TrAC
query, with the prefix cache warm (as served) and after a prefix-cache reset (worst case: every
trace is prefilled again), and the extra samples of SC@8. Repeated over seeds, at several
concurrencies (prompts submitted per call; vLLM schedules them with continuous batching).

  traj_4_4     4 checkpoints + the end of the trace, 1 greedy + 4 sampled answers each
  tf_4         teacher-forced final answer a at the same 5 positions (prefill only)
  rival_8      rival elicitation at 8 checkpoints + the end, then the teacher-forced rivals and a
               at the same positions (trac.pipeline.contrast)
  trac         traj_4_4 + tf_4 + rival_8 submitted together (what TrAC needs)
  trac_cold    trac after llm.reset_prefix_cache()
  sc_extra_7   7 further full traces per question (SC@8 = original + 7)

Writes EXP_ROOT/latency/throughput_{name}.json with per-call seconds, token counts, and the
peak KV-cache usage reported by the engine.

  python -m trac.pipeline.throughput --gen-tag math500_qwen8b_k8 --concurrency 16,256 --n 256 --reps 3
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import answer_ids, cue_ids
from trac.store import load_generation


def cuts(T, n):
    return sorted({int(round((j + 1) / (n + 1) * T)) for j in range(n)} | {T})


def timed(llm, prompts, sps):
    t = time.perf_counter()
    o = llm.generate(prompts, sps, use_tqdm=False)
    return time.perf_counter() - t, o


def trac_queries(llm, tok, bb, prefixes, answers, TokensPrompt, SamplingParams):
    """(seconds per component, seconds of the joint submission, decoded tokens, prompt tokens)."""
    from trac.pipeline.contrast import ELICIT, inner

    cc = cue_ids(tok, bb)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    samp = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=32)
    tf = lambda t: SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=1, logprob_token_ids=[t])
    traj, tfq = ([], []), ([], [])
    for (p, bd), a in zip(prefixes, answers):
        aid = answer_ids(tok, bb, a)
        for k in cuts(len(bd), 4):
            traj[0].extend([TokensPrompt(prompt_token_ids=p + bd[:k] + cc)] * 2)
            traj[1].extend([greedy, samp])
            for j, t in enumerate(aid):
                tfq[0].append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc + aid[:j]))
                tfq[1].append(tf(t))
    el = ([], [])
    for (p, bd), a in zip(prefixes, answers):
        e = M.encode(tok, bb.close + ELICIT.format(a=a))
        for k in cuts(len(bd), 8):
            el[0].append(TokensPrompt(prompt_token_ids=p + bd[:k] + e))
            el[1].append(greedy)
    sec = {}
    sec["traj_4_4"], _ = timed(llm, *traj)
    sec["tf_4"], _ = timed(llm, *tfq)
    t_el, eo = timed(llm, *el)
    cands, i = [], 0
    for (p, bd), a in zip(prefixes, answers):
        n = len(cuts(len(bd), 8))
        s = {a}
        for o in eo[i : i + n]:
            s.add(inner(tok.decode(o.outputs[0].token_ids)) or "0")
        cands.append(list(s)[:5])
        i += n
    rv = ([], [])
    for (p, bd), cs in zip(prefixes, cands):
        for c in cs:
            aid = answer_ids(tok, bb, c)
            for k in cuts(len(bd), 8):
                for j, t in enumerate(aid):
                    rv[0].append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc + aid[:j]))
                    rv[1].append(tf(t))
    t_rv, _ = timed(llm, *rv)
    sec["rival_8"] = t_el + t_rv
    allq = (traj[0] + tfq[0] + rv[0], traj[1] + tfq[1] + rv[1])
    t_joint, _ = timed(llm, *allq)  # second pass, cache warm: what serving pays
    joint = t_joint + t_el  # elicitation must precede the rival forcing
    n_dec = len(traj[0]) // 2 * 5 * 32 + len(el[0]) * 32  # upper bound on decoded answer tokens
    n_prompt = sum(len(q["prompt_token_ids"]) for q in allq[0] + el[0])
    return sec, joint, allq, el, n_dec, n_prompt


def minimal_queries(llm, tok, bb, prefixes, answers, TokensPrompt, SamplingParams):
    """Seconds of the minimal configuration: greedy answers at 4 checkpoints + the end, teacher-forced
    a at the same positions, and the rival elicited and contrasted at the end only."""
    from trac.pipeline.contrast import ELICIT, inner

    cc = cue_ids(tok, bb)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    tf = lambda t: SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=1, logprob_token_ids=[t])
    q = ([], [])
    for (p, bd), a in zip(prefixes, answers):
        aid = answer_ids(tok, bb, a)
        for k in cuts(len(bd), 4):
            q[0].append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc))
            q[1].append(greedy)
            for j, t in enumerate(aid):
                q[0].append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc + aid[:j]))
                q[1].append(tf(t))
    el = (
        [
            TokensPrompt(prompt_token_ids=p + bd + M.encode(tok, bb.close + ELICIT.format(a=a)))
            for (p, bd), a in zip(prefixes, answers)
        ],
        greedy,
    )
    t_q, _ = timed(llm, *q)
    t_el, eo = timed(llm, *el)
    rv = ([], [])
    for ((p, bd), a), o in zip(zip(prefixes, answers), eo):
        b = inner(tok.decode(o.outputs[0].token_ids)) or "0"
        bid = answer_ids(tok, bb, b)
        for j, t in enumerate(bid):
            rv[0].append(TokensPrompt(prompt_token_ids=p + bd + cc + bid[:j]))
            rv[1].append(tf(t))
    t_rv, _ = timed(llm, *rv) if rv[0] else (0.0, None)
    return t_q + t_el + t_rv, len(q[0]) + len(el[0]) + len(rv[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--concurrency", default="16,256")
    ap.add_argument("--reps", type=int, default=3, help="repetitions at concurrency < 64")
    ap.add_argument("--reps-high", type=int, default=1, help="repetitions at concurrency >= 64")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--minimal", action="store_true", help="time only the minimal configuration, without SC samples")
    a = ap.parse_args()
    name = a.name or re.sub(r"_k\d+$", "", a.gen_tag)
    if a.minimal:
        name += "_minimal"
    out = C.LATENCY / f"throughput_{name}.json"
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve("", a.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(a.gen_tag)
    llm = M.make_llm(bb, a.gpu_mem)
    timed(llm, [TokensPrompt(prompt_token_ids=rows[0]["_pids"])], SamplingParams(max_tokens=8))
    res = {"model": bb.name, "gen_tag": a.gen_tag, "runs": []}
    for B in [int(x) for x in a.concurrency.split(",")]:
        n = a.n if B >= 64 else min(a.n, 4 * B)
        for rep in range(a.reps if B < 64 else a.reps_high):
            sub = [rows[i] for i in np.random.RandomState(rep).choice(len(rows), n, replace=False)]
            for i in range(0, n, B):
                b = sub[i : i + B]
                P = [TokensPrompt(prompt_token_ids=r["_pids"]) for r in b]
                sp1 = SamplingParams(
                    n=1, temperature=0.7, top_p=0.95, max_tokens=bb.max_new, skip_special_tokens=False, seed=1000 * rep + i
                )
                rec = {"concurrency": B, "rep": rep, "size": len(b)}
                rec["primary"], outs = timed(llm, P, sp1)
                prefixes, answers = [], []
                for r, o in zip(b, outs):
                    c = o.outputs[0]
                    body, _ = M.reasoning_prefix(tok, bb, list(c.token_ids), tok.decode(c.token_ids, skip_special_tokens=False))
                    prefixes.append((r["_pids"], body))
                    answers.append((A.last_boxed(tok.decode(c.token_ids, skip_special_tokens=True)) or "0").strip() or "0")
                rec["gen_tokens"] = int(sum(len(o.outputs[0].token_ids) for o in outs))
                if a.minimal:
                    rec["minimal"], rec["n_queries_minimal"] = minimal_queries(
                        llm, tok, bb, prefixes, answers, TokensPrompt, SamplingParams
                    )
                    rec["n_queries_minimal"] /= len(b)
                    res["runs"].append(rec)
                    print(f"[thr] {rec}", flush=True)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(res, open(out, "w"), indent=1)
                    continue
                sec, joint, allq, el, n_dec, n_prompt = trac_queries(
                    llm, tok, bb, prefixes, answers, TokensPrompt, SamplingParams
                )
                rec.update(sec)
                rec["trac"] = joint
                rec["probe_decode_tokens_max"] = n_dec
                rec["probe_prompt_tokens"] = n_prompt
                rec["n_queries"] = len(allq[0]) + len(el[0])
                llm.reset_prefix_cache()
                t_el, _ = timed(llm, *el)
                t_rest, _ = timed(llm, *allq)
                rec["trac_cold"] = t_el + t_rest
                spx = SamplingParams(n=7, temperature=0.7, top_p=0.95, max_tokens=bb.max_new, seed=1000 * rep + i + 7)
                rec["sc_extra_7"], so = timed(llm, P, spx)
                rec["sc_tokens"] = int(sum(len(x.token_ids) for o in so for x in o.outputs))
                res["runs"].append(rec)
                print(f"[thr] {rec}", flush=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(out, "w"), indent=1)
    print(f"[thr] -> {out}")


if __name__ == "__main__":
    main()
