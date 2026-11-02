"""Sampling latency audit on one GPU (fresh generations, batches of --batch concurrent queries, prefix
caching on); every timing is relative to the original generation of the same batch. It supplies the
SC@2 latency of the paper; trac.pipeline.throughput audits every TrAC query.

  primary       one full trace (the 1x reference)
  end           one short answer decode right after the original trace (prefix cached)
  sc_k          k full traces (n = k), k in {2, 4, 8}
  end_nocache   the same answer decode on a second engine with prefix caching disabled

  python -m trac.pipeline.latency --gen-tag math500_qwen8b_k8   -> EXP_ROOT/latency/sampling_math500_qwen8b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import cue_ids
from trac.store import load_generation


def timed(llm, prompts, sps):
    t = time.perf_counter()
    o = llm.generate(prompts, sps, use_tqdm=False)
    return time.perf_counter() - t, o


def audit_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams, max_tokens):
    cc = cue_ids(tok, bb)
    spp = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    pp = []
    for r, o in zip(b, outs):
        c = o.outputs[0]
        body, _ = M.reasoning_prefix(tok, bb, list(c.token_ids), tok.decode(c.token_ids, skip_special_tokens=False))
        pp.append(TokensPrompt(prompt_token_ids=r["_pids"] + body + cc))
    rec["end"], _ = timed(llm, pp, spp)
    P = [TokensPrompt(prompt_token_ids=r["_pids"]) for r in b]
    for k in (2, 4, 8):
        rec[f"sc_{k}"], _ = timed(llm, P, SamplingParams(n=k, temperature=0.7, top_p=0.95, max_tokens=max_tokens))
    return pp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True, help="questions (prompts) come from this generation cache")
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--name", default="", help="audit name (default: gen-tag without its _kK suffix)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = backbone default budget")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()
    name = args.name or re.sub(r"_k\d+$", "", args.gen_tag)
    out = C.LATENCY / f"sampling_{name}.json"
    if out.exists():
        print(f"[lat] exists {out}")
        return

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
    for b in [rows[i : i + args.batch] for i in range(0, len(rows), args.batch)]:
        rec = {}
        rec["primary"], outs = timed(llm, [TokensPrompt(prompt_token_ids=r["_pids"]) for r in b], sp1)
        probe_prompts.append(audit_batch(llm, tok, bb, b, outs, rec, TokensPrompt, SamplingParams, max_tokens))
        rec["gen_tokens"] = int(sum(len(o.outputs[0].token_ids) for o in outs))
        res["batches"].append(rec)
        print(f"[lat] batch: {rec}", flush=True)
    del llm  # the same answer decodes without prefix caching
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    llm = M.make_llm(bb, args.gpu_mem, enable_prefix_caching=False)
    spp = SamplingParams(n=1, temperature=0.0, max_tokens=32)
    for rec, pp in zip(res["batches"], probe_prompts):
        rec["end_nocache"], _ = timed(llm, pp, spp)
    tot = {k: float(np.sum([b[k] for b in res["batches"]])) for k in res["batches"][0] if k != "gen_tokens"}
    res["relative"] = {
        "end_cached": (tot["primary"] + tot["end"]) / tot["primary"],
        "end_nocache": (tot["primary"] + tot["end_nocache"]) / tot["primary"],
        **{f"sc_{k}": tot[f"sc_{k}"] / tot["primary"] for k in (2, 4, 8)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[lat] relative latency: {res['relative']} -> {out}")


if __name__ == "__main__":
    main()
