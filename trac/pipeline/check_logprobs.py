"""Numerical sanity check: vLLM-stored chosen-token logprobs vs. an HF teacher-forced
recomputation of the same tokens (chunked prefill). Validates token alignment of the
stored traces and, for interleaved sliding-window models (Olmo-3), that long-context
positions (> sliding window) agree as well as short-context ones.

  python -m trac.pipeline.check_logprobs --gen-tag smoke_olmo7b --n 3
"""
from __future__ import annotations

import argparse

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.config import model_path
from trac.store import load_generation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-tokens", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache

    bb = M.resolve(args.model, args.gen_tag)
    rows = load_generation(args.gen_tag)
    z = np.load(C.GEN / f"{args.gen_tag}.tok.npz")
    oo = np.concatenate([[0], np.cumsum(z["out_len"])])
    idx = [i for i, r in enumerate(rows) if len(r["_oids"]) >= args.min_tokens][: args.n]
    if not idx:
        idx = sorted(range(len(rows)), key=lambda i: -len(rows[i]["_oids"]))[: args.n]
    model = AutoModelForCausalLM.from_pretrained(model_path(bb.name), dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True).eval()
    for i in idx:
        r = rows[i]
        ids = r["_pids"] + r["_oids"]
        P = len(r["_pids"])
        hf = np.zeros(len(r["_oids"]))
        cache = DynamicCache(config=model.config)
        with torch.inference_mode():
            for s0 in range(0, len(ids) - 1, args.chunk):
                s1 = min(len(ids) - 1, s0 + args.chunk)
                o = model(torch.tensor([ids[s0:s1]], device="cuda:0"), past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                lp = torch.log_softmax(o.logits[0].float(), -1)
                tgt = torch.tensor(ids[s0 + 1:s1 + 1], device="cuda:0")
                g = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
                for k, pos in enumerate(range(s0 + 1, s1 + 1)):
                    if pos >= P:
                        hf[pos - P] = g[k]
                del o, lp
        vl = z["chosen_lp"][oo[i]:oo[i + 1]].astype(float)
        d = np.abs(hf - vl)
        ctx = P + np.arange(len(vl))
        near, far = ctx < 4096, ctx >= 4096
        print(f"[chk] {r['id']}: n_out={len(vl)}  |dlp| mean ctx<4096={d[near].mean():.4f}  "
              f"ctx>=4096={d[far].mean() if far.any() else float('nan'):.4f}  "
              f"corr={np.corrcoef(hf, vl)[0, 1]:.4f}  mean lp vllm={vl.mean():.4f} hf={hf.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
