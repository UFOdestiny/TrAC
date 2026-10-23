"""Hidden-state features for the matched-supervision white-box baselines.

For every stored original trace, two teacher-forced prefill passes (HF transformers,
no generation) record the residual stream at layers round(f*L), f in {0.5, 0.75, 1.0}:
  pass 1  prompt + full original output
          end_reason   last token of the reasoning body
          answer_end   last token of the returned answer section
          mean_reason  mean over reasoning-body tokens
  pass 2  prompt + reasoning body + close + PCE cue
          cue          last cue token (the position PCE decodes from)

Writes EXP_ROOT/hidden/{tag}.npz: H (N, 4 positions, 3 layers, d) float16,
ids, positions, layers. These feed a logistic probe trained on the same folds and
labels as TrAC (HS-probe), the Liu et al. (2024) style supervised estimator, and
the TrAC + HS fusion.

  python -m trac.pipeline.hidden_states --gen-tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.config import model_path
from trac.pipeline.probes import cue_ids
from trac.store import load_generation

POSITIONS = ["end_reason", "answer_end", "mean_reason", "cue"]
FRACS = [0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk length")
    args = ap.parse_args()

    out = C.HIDDEN / f"{args.gen_tag}.npz"
    if out.exists():
        print(f"[hs] exists {out}"); return
    import torch
    from transformers import AutoModelForCausalLM

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    if args.limit:
        rows = rows[: args.limit]
    model = AutoModelForCausalLM.from_pretrained(model_path(bb.name), dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    L = model.config.num_hidden_layers
    layers = [int(round(f * L)) for f in FRACS]
    close_cue = cue_ids(tok, bb)
    d = model.config.hidden_size
    H = np.zeros((len(rows), len(POSITIONS), len(layers), d), np.float16)

    def chunked(ids, points, mean_lo, mean_hi):
        """Chunked prefill with a KV cache (bounded attention memory, incl. eager
        attention for sink models). Returns {point: (layers, d)} and the mean over
        [mean_lo, mean_hi) per layer."""
        from transformers import DynamicCache
        cache = DynamicCache(config=model.config)
        got, msum = {}, torch.zeros(len(layers), d, device="cuda:0", dtype=torch.float32)
        for s0 in range(0, len(ids), args.chunk):
            s1 = min(len(ids), s0 + args.chunk)
            o = model(torch.tensor([ids[s0:s1]], device="cuda:0"), past_key_values=cache, use_cache=True,
                      output_hidden_states=True, logits_to_keep=1)
            cache = o.past_key_values
            hs = torch.stack([o.hidden_states[l][0] for l in layers])          # (layers, chunk, d)
            for pt in points:
                if s0 <= pt < s1:
                    got[pt] = hs[:, pt - s0].float().cpu().numpy()
            lo, hi = max(mean_lo, s0), min(mean_hi, s1)
            if hi > lo:
                msum += hs[:, lo - s0:hi - s0].float().sum(1)
            del o, hs
        return got, (msum / max(1, mean_hi - mean_lo)).cpu().numpy()

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["_oids"]))
    t0 = time.time()
    with torch.inference_mode():
        for n, i in enumerate(order):
            r = rows[i]
            P = len(r["_pids"])
            body, _ = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
            nb = max(1, min(len(body), len(r["_oids"])))
            seq1 = r["_pids"] + r["_oids"]
            e_r, a_e = P + nb - 1, len(seq1) - 1
            got, mean = chunked(seq1, [e_r, a_e], P, P + nb)
            H[i, 0], H[i, 1], H[i, 2] = got[e_r], got[a_e], mean
            seq2 = r["_pids"] + body + close_cue
            got, _ = chunked(seq2, [len(seq2) - 1], 0, 0)
            H[i, 3] = got[len(seq2) - 1]
            if (n + 1) % 200 == 0:
                print(f"[hs] {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, H=H, ids=np.array([r["id"] for r in rows]), positions=np.array(POSITIONS),
             layers=np.array(layers))
    print(f"[hs] saved {H.shape} -> {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
