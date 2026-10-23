"""Answer trajectories of the K-1 extra samples (for the trajectory-weighted vote and the
within-question comparison).

For every stored extra sample k of a question, its reasoning body is rebuilt by encoding the
stored raw output after the question's prompt ids (re-encoded text, so the prefix is a faithful
re-reading rather than the exact sampled token sequence). The model then re-elicits one greedy
answer after close + cue 0 at C evenly spaced checkpoints (moved back to a line break) and at the
completed prefix, and we record each answer's agreement with the sample's own final answer.

Output: EXP_ROOT/sample_trajectory/{tag}.json.gz, per question one record per extra sample
  {k, T, closed, agree_own: [C + 1 bools], answers: [C + 1]}

  python -m trac.pipeline.sample_trajectory --gen-tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import time

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import NewlineTokens, checkpoints, cue_ids, even_fractions, token_chunks
from trac.store import load_generation, write_json_gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--checkpoints", type=int, default=8)
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--chunk-tokens", type=int, default=400_000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = C.SAMPLE_TRAJECTORY / f"{args.gen_tag}.json.gz"
    if out.exists():
        print(f"[ts] exists {out}"); return

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    if args.limit:
        rows = rows[:args.limit]
    cc = cue_ids(tok, bb)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, skip_special_tokens=False)
    has_nl = NewlineTokens(tok)
    fracs = even_fractions(args.checkpoints)

    items = []
    for i, r in enumerate(rows):
        for k, s in enumerate(r["samples"]):
            body, ok = M.reasoning_prefix(tok, bb, M.encode(tok, s["raw"]), s["raw"])
            prefix = r["_pids"] + body
            if len(prefix) + args.probe_tokens + 64 > bb.max_model_len:
                continue
            items.append({"i": i, "k": k, "prefix": prefix, "cks": checkpoints(body, fracs, has_nl) + [len(body)],
                          "P": len(r["_pids"]), "closed": ok})

    llm = M.make_llm(bb, args.gpu_mem)
    res = {}
    t0 = time.time()
    chunks = token_chunks(items, args.chunk_tokens)
    for ci, ch in enumerate(chunks):
        # completed prefix first (warms the cache), then the earlier checkpoints
        for phase in ("end", "ck"):
            reqs, keys = [], []
            for n, it in enumerate(ch):
                cuts = it["cks"][-1:] if phase == "end" else it["cks"][:-1]
                for c in cuts:
                    reqs.append(TokensPrompt(prompt_token_ids=it["prefix"][:it["P"] + c] + cc)); keys.append((n, c))
            outs = llm.generate(reqs, greedy, use_tqdm=False) if reqs else []
            for (n, c), o in zip(keys, outs):
                it = ch[n]; r = rows[it["i"]]; s = r["samples"][it["k"]]
                ans = A.extract_head(r, tok.decode(o.outputs[0].token_ids, skip_special_tokens=False))
                it.setdefault("ans", {})[c] = (ans, bool(A.equiv(r, ans, s["answer"])))
        for it in ch:
            rec = {"k": it["k"], "T": it["cks"][-1], "closed": it["closed"],
                   "agree_own": [it["ans"][c][1] for c in it["cks"]],
                   "answers": [it["ans"][c][0] for c in it["cks"]]}
            res.setdefault(rows[it["i"]]["id"], []).append(rec)
        print(f"[ts] {args.gen_tag} chunk {ci + 1}/{len(chunks)} {(time.time() - t0) / 60:.1f} min", flush=True)
    write_json_gz(out, [{"id": r["id"], "samples": res.get(r["id"], [])} for r in rows])
    print(f"[ts] saved -> {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
