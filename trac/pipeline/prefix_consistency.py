"""Prefix-consistency baseline (Iwase et al., 2026) for all K traces of every question.

Each stored trace y (the original response and the K-1 extra samples) is cut after its first
ceil(tau * |y|) output tokens (tau = 0.75), and the remainder is regenerated --regens times
with the generation sampling settings (T = 0.7, top-p 0.95, the backbone's remaining budget).
The final answer of each regenerated response is extracted exactly as in trac.pipeline.generate.
Prefix consistency of a trace = share of {original answer} + {regenerated answers} that equal
the original answer. Regeneration cost (new tokens) is recorded per trace.

Output: EXP_ROOT/prefix_consistency/{tag}.json.gz

  python -m trac.pipeline.prefix_consistency --gen-tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import math
import time

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.store import load_generation, write_json_gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--regens", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=256, help="questions per generate call")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--primary-only", action="store_true", help="only the original response (no extra samples)")
    args = ap.parse_args()
    out = C.PREFIX_CONSISTENCY / f"{args.gen_tag}.json.gz"
    if out.exists():
        print(f"[pc] exists {out}"); return

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    if args.limit:
        rows = rows[:args.limit]
    llm = M.make_llm(bb, args.gpu_mem, args.seed)
    res = []
    t0 = time.time()
    for c0 in range(0, len(rows), args.chunk):
        ch = rows[c0:c0 + args.chunk]
        reqs, sps, keys = [], [], []
        for qi, r in enumerate(ch):
            traces = [(0, r["_oids"], r["answer"])] + [
                (k + 1, M.encode(tok, s["raw"]), s["answer"]) for k, s in enumerate(r["samples"])
                if not args.primary_only]
            for k, ids, _ in traces:
                cut = math.ceil(args.tau * len(ids))
                budget = min(bb.max_new - cut, bb.max_model_len - len(r["_pids"]) - cut - 8)
                if cut < 1 or budget < 16:
                    continue
                reqs.append(TokensPrompt(prompt_token_ids=r["_pids"] + ids[:cut]))
                sps.append(SamplingParams(n=args.regens, temperature=0.7, top_p=0.95, max_tokens=budget,
                                          skip_special_tokens=False, seed=args.seed + k))
                keys.append((qi, k, ids[:cut]))
        outs = llm.generate(reqs, sps, use_tqdm=False)
        recs = [{"id": r["id"], "traces": {}} for r in ch]
        for (qi, k, pre), o in zip(keys, outs):
            r = ch[qi]
            orig = r["answer"] if k == 0 else r["samples"][k - 1]["answer"]
            ans, ntok = [], []
            for comp in o.outputs:
                raw = tok.decode(list(pre) + list(comp.token_ids), skip_special_tokens=False)
                final = M.final_for(bb, raw, comp.finish_reason)
                ans.append(A.extract_answer(r, final)); ntok.append(len(comp.token_ids))
            same = [bool(A.equiv(r, a, orig)) for a in ans]
            recs[qi]["traces"][str(k)] = {"pc": (1 + sum(same)) / (1 + len(same)), "same": same,
                                          "regen_tokens": ntok, "prefix_tokens": len(pre)}
        res.extend(recs)
        print(f"[pc] {args.gen_tag} {min(c0 + args.chunk, len(rows))}/{len(rows)} {(time.time() - t0) / 60:.1f} min",
              flush=True)
    write_json_gz(out, res)
    print(f"[pc] saved -> {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
