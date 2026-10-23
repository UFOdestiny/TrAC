"""Re-elicitation along one stored reasoning trace (one vLLM session per tag).

For each stored original trace with reasoning body r_1..r_T and returned answer a:
  ck   at checkpoints t_j (fractions j / (C + 1) of the body, j = 1..C, moved back to a
       line break): one greedy and S sampled re-elicited answers after close + cue 0, and
       the teacher-forced log-likelihood of every token of a at the same position
  end  the same at the completed prefix (t = T)

All checkpoint prefixes are prefixes of the same token sequence, so after one prefill of
the trace every probe reads the prefix cache; traces are processed in chunks whose total
length fits the KV cache, with the completed prefix decoded first to warm it. Teacher
forcing uses one-token requests that score the next answer token (logprob_token_ids),
which keeps prefix caching on (prompt_logprobs would disable it).

Output: EXP_ROOT/trajectory/{tag}.json.gz, one record per question
  {id, T, closed, ck: [{k, ans, surf, agree, tok_lps, samp: {ans, agree}, tf}], end: {...}}

  python -m trac.pipeline.trajectory --gen-tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import time

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import (
    NewlineTokens,
    answer_ids,
    answer_span,
    checkpoints,
    chosen_logprobs,
    cue_ids,
    even_fractions,
    score_token_params,
    scored_logprob,
    token_chunks,
)
from trac.store import load_generation, write_json_gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--checkpoints", type=int, default=20, help="C evenly spaced checkpoints")
    ap.add_argument("--samples", type=int, default=4, help="S sampled re-elicitations per checkpoint")
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--chunk-tokens", type=int, default=300_000, help="trace tokens per cache chunk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    fracs = even_fractions(args.checkpoints, decimals=4)

    out = C.TRAJECTORY / f"{args.gen_tag}.json.gz"
    if out.exists():
        print(f"[traj] exists {out}"); return

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    if args.limit:
        rows = rows[:args.limit]
    cc = cue_ids(tok, bb)
    has_nl = NewlineTokens(tok)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1,
                            flat_logprobs=True, skip_special_tokens=False)
    samp = SamplingParams(n=max(1, args.samples), temperature=0.7, top_p=0.95, max_tokens=args.probe_tokens,
                          logprobs=1, flat_logprobs=True, skip_special_tokens=False, seed=args.seed)

    def comp_rec(r, comp):
        ids = list(comp.token_ids)
        text = tok.decode(ids, skip_special_tokens=False)
        ans = A.extract_head(r, text)
        return {"ans": ans, "surf": A.boxed_continuation(text),
                "agree": bool(A.equiv(r, ans, r["answer"])), "tok_lps": chosen_logprobs(comp)[:answer_span(tok, ids)]}

    items = []
    for i, r in enumerate(rows):
        body, ok = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        a = A.surface_answer(r)
        items.append({"i": i, "prefix": r["_pids"] + body, "P": len(r["_pids"]), "T": len(body),
                      "cks": checkpoints(body, fracs, has_nl), "closed": ok,
                      "a_ids": answer_ids(tok, bb, a) if a is not None else None})

    fits = lambda ids: len(ids) + args.probe_tokens + 64 <= bb.max_model_len
    llm = M.make_llm(bb, args.gpu_mem, args.seed)

    def run(reqs):
        if not reqs:
            return []
        return llm.generate([TokensPrompt(prompt_token_ids=x[0]) for x in reqs], [x[1] for x in reqs],
                            use_tqdm=False)

    res = {}
    t0 = time.time()
    chunks = token_chunks(items, args.chunk_tokens)
    for ci, ch in enumerate(chunks):
        rec = {it["i"]: {"id": rows[it["i"]]["id"], "T": it["T"], "closed": it["closed"], "ck": [], "end": None}
               for it in ch}
        # phase A: greedy answer at the completed prefix (warms the cache)
        live = [it for it in ch if fits(it["prefix"])]
        for it, o in zip(live, run([(it["prefix"] + cc, greedy) for it in live])):
            rec[it["i"]]["end"] = comp_rec(rows[it["i"]], o.outputs[0])
        # phase B: checkpoints (greedy + sampled), sampled at the end, teacher-forced a everywhere
        reqs, keys = [], []
        for it in live:
            i = it["i"]
            ctxs = [("ck", n, it["prefix"][:it["P"] + k]) for n, k in enumerate(it["cks"])]
            ctxs.append(("end", 0, it["prefix"]))
            for kind, n, ctx in ctxs:
                if kind == "ck":
                    reqs.append((ctx + cc, greedy)); keys.append((i, kind, n, "g", -1))
                if args.samples:
                    reqs.append((ctx + cc, samp)); keys.append((i, kind, n, "s", -1))
                for j, t in enumerate(it["a_ids"] or []):
                    reqs.append((ctx + cc + it["a_ids"][:j], score_token_params(t))); keys.append((i, kind, n, "tf", j))
        tf, samp_res = {}, {}
        for (i, kind, n, what, j), o in zip(keys, run(reqs)):
            if what == "s":
                sm = [comp_rec(rows[i], c) for c in o.outputs]
                samp_res[(i, kind, n)] = {"ans": [x["ans"] for x in sm], "agree": [x["agree"] for x in sm]}
            elif what == "g":
                x = comp_rec(rows[i], o.outputs[0])
                x["k"] = items[i]["cks"][n]
                rec[i]["ck"].append(x)
            else:
                tf.setdefault((i, kind, n), {})[j] = scored_logprob(o, items[i]["a_ids"][j])
        for (i, kind, n), d in samp_res.items():
            if kind == "ck":
                rec[i]["ck"][n]["samp"] = d
            elif rec[i]["end"] is not None:
                rec[i]["end"]["samp"] = d
        for (i, kind, n), d in tf.items():
            v = [d.get(j) for j in range(len(items[i]["a_ids"]))]
            if kind == "ck":
                rec[i]["ck"][n]["tf"] = v
            else:
                rec[i]["end"]["tf"] = v
        res.update(rec)
        print(f"[traj] {args.gen_tag} chunk {ci + 1}/{len(chunks)} ({len(ch)} traces) "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    write_json_gz(out, [res[i] for i in sorted(res)])
    print(f"[traj] saved -> {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
