"""Runner-up answers along one stored reasoning trace (one vLLM session per tag).

Reads the intermediate-answer cache of the same tag (its checkpoints and intermediate answers) and,
on the same prefix-cached trace, adds two queries:

  elic   runner-up elicitation: at R positions (evenly spaced checkpoints + the end of the trace),
         greedy decode after the end-of-reasoning marker + "The final answer is \\boxed{a}. If this answer
         were wrong, the most likely alternative answer would be \\boxed{"
  tf     the teacher-forced log-likelihood of every token of each candidate (the final answer a,
         the most frequent other intermediate answers, and the elicited runner-ups) after the
         answer cue, at the same R positions

Candidates are compared with the typed equivalence of rewind.answers; labels never enter.

Output: EXP_ROOT/runner_up/{tag}.json.gz (RUNNER_UP), one record per question
  {id, pos: [k...], cands: [{surf, src: a | intermediate | elic}], tf: [[[lp per token] per position] per candidate],
   elic: [{k, surf, ans, eq_a, tok_lps}]}

  python -m rewind.pipeline.runner_up --gen-tag math500_qwen8b_k8
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from rewind import answers as A
from rewind import backbones as M
from rewind import config as C
from rewind.pipeline.queries import (
    answer_ids,
    answer_span,
    chosen_logprobs,
    cue_ids,
    score_token_params,
    scored_logprob,
    token_chunks,
)
from rewind.store import load_generation, read_json_gz, write_json_gz

RUNNER_UP_PROMPT = "The final answer is \\boxed{{{a}}}. If this answer were wrong, the most likely alternative answer would be \\boxed{{"


def inner(head):
    """Raw text inside a boxed continuation, up to its matching closing brace."""
    depth, out = 1, []
    for ch in head or "":
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            break
        out.append(ch)
    return "".join(out).strip()


def thin_positions(cks, n):
    """n evenly spaced checkpoint lengths (from the intermediate-answer checkpoints)."""
    if len(cks) <= n:
        return list(cks)
    step = (len(cks) - 1) / max(1, n - 1)
    return sorted({cks[int(round(j * step))] for j in range(n)})


def frequent_other_answers(row, rec, a, top):
    """Most frequent intermediate answers that differ from a, with a surface form each."""
    count, surf = Counter(), {}
    for c in list(rec["ck"]) + [rec["end"] or {}]:
        pairs = [(c.get("ans"), c.get("surf"))] + [(x, None) for x in (c.get("samp") or {}).get("ans", [])]
        for ans, s in pairs:
            if ans is None or A.equiv(row, ans, a):
                continue
            key = str(ans)
            count[key] += 1
            if s and key not in surf and inner(s):
                surf[key] = inner(s)
    return [surf.get(k, k) for k, _ in count.most_common(top)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--positions", type=int, default=8, help="checkpoints queried (plus the end)")
    ap.add_argument("--frequent", type=int, default=3, help="most frequent other intermediate answers teacher-forced")
    ap.add_argument("--query-tokens", type=int, default=32)
    ap.add_argument("--chunk-tokens", type=int, default=300_000, help="trace tokens per cache chunk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = C.RUNNER_UP / f"{args.gen_tag}.json.gz"
    if out.exists():
        print(f"[runner-up] exists {out}")
        return
    intermediate = {r["id"]: r for r in read_json_gz(C.INTERMEDIATE_ANSWERS / f"{args.gen_tag}.json.gz")}

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    if args.limit:
        rows = rows[: args.limit]
    cc = cue_ids(tok, bb)
    greedy = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.query_tokens, logprobs=1, flat_logprobs=True, skip_special_tokens=False
    )

    items = []
    for i, r in enumerate(rows):
        tr = intermediate.get(r["id"])
        a = A.surface_answer(r)
        body, _ = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        cks = [c["k"] for c in tr["ck"]] if tr else []
        items.append(
            {
                "i": i,
                "prefix": r["_pids"] + body,
                "P": len(r["_pids"]),
                "a": a,
                "pos": thin_positions(cks, args.positions) + [len(body)],
                "frequent": frequent_other_answers(r, tr, r["answer"], args.frequent) if (tr and a) else [],
            }
        )

    fits = lambda ids: len(ids) + args.query_tokens + 128 <= bb.max_model_len
    llm = M.make_llm(bb, args.gpu_mem, args.seed)

    def run(reqs):
        if not reqs:
            return []
        return llm.generate([TokensPrompt(prompt_token_ids=x[0]) for x in reqs], [x[1] for x in reqs], use_tqdm=False)

    res = {}
    t0 = time.time()
    chunks = token_chunks(items, args.chunk_tokens)
    for ci, ch in enumerate(chunks):
        live = [it for it in ch if it["a"] is not None and fits(it["prefix"])]
        rec = {it["i"]: {"id": rows[it["i"]]["id"], "pos": it["pos"], "cands": [], "tf": [], "elic": []} for it in ch}
        # phase A: runner-up elicitation at every position (the end first, which warms the cache)
        reqs, keys = [], []
        for it in live:
            el = M.encode(tok, bb.close + RUNNER_UP_PROMPT.format(a=it["a"]))
            for k in reversed(it["pos"]):
                reqs.append((it["prefix"][: it["P"] + k] + el, greedy))
                keys.append((it["i"], k))
        for (i, k), o in zip(keys, run(reqs)):
            comp = o.outputs[0]
            ids = list(comp.token_ids)
            text = tok.decode(ids, skip_special_tokens=False)
            ans = A.extract_head(rows[i], text)
            rec[i]["elic"].append(
                {
                    "k": k,
                    "surf": inner(text),
                    "ans": ans,
                    "eq_a": bool(A.equiv(rows[i], ans, rows[i]["answer"])),
                    "tok_lps": chosen_logprobs(comp)[: answer_span(tok, ids)],
                }
            )
        # candidates: a, the most frequent other intermediate answers, then elicited runner-ups not yet covered
        for it in live:
            row, cands = rows[it["i"]], [{"surf": it["a"], "src": "a"}]
            seen = [row["answer"]]
            for s in it["frequent"]:
                cands.append({"surf": s, "src": "intermediate"})
                seen.append(A.extract_head(row, s + "}") or s)
            for e in sorted(rec[it["i"]]["elic"], key=lambda e: -e["k"]):
                if e["ans"] is not None and e["surf"] and not any(A.equiv(row, e["ans"], x) for x in seen):
                    cands.append({"surf": e["surf"], "src": "elic"})
                    seen.append(e["ans"])
            for c in cands:
                c["ids"] = answer_ids(tok, bb, c["surf"])
            rec[it["i"]]["cands"] = cands
        # phase B: teacher forcing of every candidate
        reqs, keys = [], []
        for it in live:
            i = it["i"]
            for n, k in enumerate(it["pos"]):
                ctx = it["prefix"][: it["P"] + k] + cc
                for ci_, c in enumerate(rec[i]["cands"]):
                    for j, t in enumerate(c["ids"]):
                        reqs.append((ctx + c["ids"][:j], score_token_params(t)))
                        keys.append((i, ci_, n, j))
        tf = {}
        for (i, a_, b_, j), o in zip(keys, run(reqs)):
            tf.setdefault((i, a_, b_), {})[j] = scored_logprob(o, rec[i]["cands"][a_]["ids"][j])
        for it in live:
            i = it["i"]
            rec[i]["tf"] = [
                [[tf.get((i, ci_, n), {}).get(j) for j in range(len(c["ids"]))] for n in range(len(it["pos"]))]
                for ci_, c in enumerate(rec[i]["cands"])
            ]
            for c in rec[i]["cands"]:
                c.pop("ids", None)
            rec[i]["elic"].sort(key=lambda e: e["k"])
        res.update(rec)
        print(
            f"[runner-up] {args.gen_tag} chunk {ci + 1}/{len(chunks)} ({len(ch)} traces) {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )

    write_json_gz(out, [res[i] for i in sorted(res)])
    print(f"[runner-up] saved -> {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
