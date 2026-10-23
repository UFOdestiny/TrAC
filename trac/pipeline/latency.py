"""Wall-clock latency audits on one GPU (fresh generations, batches of --batch concurrent
queries, prefix caching on). Every timing is relative to the original generation.

--mode endpoint      writes EXP_ROOT/latency/endpoint_{name}.json
  primary      one full trace (the 1x reference)
  pce          one short cue-0 answer decode right after the original trace (prefix cached)
  sc_k         k full traces (n=k), k in {2, 4, 8}
  pce_nocache  the same probe on a second engine with prefix caching disabled

--mode trajectory    writes EXP_ROOT/latency/trajectory_{name}.json
  end          one greedy answer at the completed prefix
  traj_C_S     C evenly spaced checkpoints + the completed prefix, each with one greedy and
               S sampled answers (every query TrAC-0 needs)
  full_C_S     every query of the supervised features: traj_C_S plus the teacher-forced final
               answer at each checkpoint, and at the completed prefix the two paraphrased cues
               and R = 8 sampled re-elicitations (C = 0: the completed-prefix features alone)

  python -m trac.pipeline.latency --mode trajectory --gen-tag math500_qwen8b_k8   # -> trajectory_math500_qwen8b.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import answer_ids, cue_ids
from trac.prompts import CUES
from trac.store import load_generation

TRAJ_CONFIGS = [(4, 4), (8, 4), (20, 0), (20, 4)]
FULL_CONFIGS = [(0, 0), (4, 4), (8, 4), (20, 0), (20, 4)]
REPEATS = 8


def timed(llm, prompts, sps):
    t = time.perf_counter()
    o = llm.generate(prompts, sps, use_tqdm=False)
    return time.perf_counter() - t, o


def endpoint_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams, max_tokens):
    cc = cue_ids(tok, bb)
    spp = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    pp = []
    for r, o in zip(b, outs):
        c = o.outputs[0]
        body, _ = M.reasoning_prefix(tok, bb, list(c.token_ids), tok.decode(c.token_ids, skip_special_tokens=False))
        pp.append(TokensPrompt(prompt_token_ids=r["_pids"] + body + cc))
    rec["pce"], _ = timed(llm, pp, spp)
    P = [TokensPrompt(prompt_token_ids=r["_pids"]) for r in b]
    for k in (2, 4, 8):
        rec[f"sc_{k}"], _ = timed(llm, P, SamplingParams(n=k, temperature=0.7, top_p=0.95, max_tokens=max_tokens))
    return pp


def trajectory_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams):
    cc = cue_ids(tok, bb)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    prefixes = []
    for r, o in zip(b, outs):
        c = o.outputs[0]
        body, _ = M.reasoning_prefix(tok, bb, list(c.token_ids), tok.decode(c.token_ids, skip_special_tokens=False))
        prefixes.append((r["_pids"], body))
    rec["end"], _ = timed(llm, [TokensPrompt(prompt_token_ids=p + bd + cc) for p, bd in prefixes], greedy)

    def cuts(T, n):
        return sorted({int(round((j + 1) / (n + 1) * T)) for j in range(n)} | {T})

    for n, S in TRAJ_CONFIGS:
        P, SP = [], []
        samp = SamplingParams(n=max(S, 1), temperature=0.7, top_p=0.95, max_tokens=32)
        for p, bd in prefixes:
            for k in cuts(len(bd), n):
                P.append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc)); SP.append(greedy)
                if S:
                    P.append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc)); SP.append(samp)
        rec[f"traj_{n}_{S}"], _ = timed(llm, P, SP)
    paraphrases = [cue_ids(tok, bb, c) for c in range(1, len(CUES))]
    for n, S in FULL_CONFIGS:
        P, SP = [], []
        samp = SamplingParams(n=max(S, 1), temperature=0.7, top_p=0.95, max_tokens=32)
        rep = SamplingParams(n=REPEATS, temperature=0.7, top_p=0.95, max_tokens=32)
        for (p, bd), o in zip(prefixes, outs):
            text = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True)
            aid = answer_ids(tok, bb, (A.last_boxed(text) or "0").strip() or "0")
            for k in cuts(len(bd), n):
                P.append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc)); SP.append(greedy)
                if S:
                    P.append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc)); SP.append(samp)
                for j, t in enumerate(aid):
                    P.append(TokensPrompt(prompt_token_ids=p + bd[:k] + cc + aid[:j]))
                    SP.append(SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=1, logprob_token_ids=[t]))
            for c in paraphrases:
                P.append(TokensPrompt(prompt_token_ids=p + bd + c)); SP.append(greedy)
            P.append(TokensPrompt(prompt_token_ids=p + bd + cc)); SP.append(rep)
        rec[f"full_{n}_{S}"], _ = timed(llm, P, SP)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["endpoint", "trajectory"], required=True)
    ap.add_argument("--gen-tag", required=True, help="questions (prompts) come from this generation cache")
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--name", default="", help="audit name (default: gen-tag without its _kK suffix)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = backbone default budget")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()
    name = args.name or re.sub(r"_k\d+$", "", args.gen_tag)
    out = C.LATENCY / f"{args.mode}_{name}.json"
    if out.exists():
        print(f"[lat] exists {out}"); return

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)[: args.n]
    max_tokens = args.max_tokens or bb.max_new
    sp1 = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=max_tokens, skip_special_tokens=False)
    llm = M.make_llm(bb, args.gpu_mem)
    timed(llm, [TokensPrompt(prompt_token_ids=rows[0]["_pids"])], SamplingParams(max_tokens=8))  # warmup
    res = {"model": bb.name, "n": len(rows), "batch": args.batch, "batches": []}
    probe_prompts = []
    for b in [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]:
        rec = {}
        rec["primary"], outs = timed(llm, [TokensPrompt(prompt_token_ids=r["_pids"]) for r in b], sp1)
        if args.mode == "endpoint":
            probe_prompts.append(endpoint_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams, max_tokens))
        else:
            trajectory_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams)
        rec["gen_tokens"] = int(sum(len(o.outputs[0].token_ids) for o in outs))
        res["batches"].append(rec)
        print(f"[lat] batch: {rec}", flush=True)
    if args.mode == "endpoint":                    # the same probes without prefix caching
        del llm
        import torch
        gc.collect(); torch.cuda.empty_cache()
        llm = M.make_llm(bb, args.gpu_mem, enable_prefix_caching=False)
        spp = SamplingParams(n=1, temperature=0.0, max_tokens=32)
        for rec, pp in zip(res["batches"], probe_prompts):
            rec["pce_nocache"], _ = timed(llm, pp, spp)
    tot = {k: float(np.sum([b[k] for b in res["batches"]])) for k in res["batches"][0] if k != "gen_tokens"}
    if args.mode == "endpoint":
        res["relative"] = {"trac_cached": (tot["primary"] + tot["pce"]) / tot["primary"],
                           "trac_nocache": (tot["primary"] + tot["pce_nocache"]) / tot["primary"],
                           **{f"sc_{k}": tot[f"sc_{k}"] / tot["primary"] for k in (2, 4, 8)}}
    else:
        res["relative"] = {k: (tot["primary"] + tot[k]) / tot["primary"] for k in tot if k != "primary"}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[lat] relative latency: {res['relative']} -> {out}")


if __name__ == "__main__":
    main()
