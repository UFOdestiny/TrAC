"""Generation: one original trace (+ K-1 extra samples) per question, token-exact.

Writes, per tag under EXP_ROOT/gen/:
  {tag}.json.gz   one record per question: question/gold metadata, the original trace's
                  raw output, returned answer section, extracted answer, label, finish
                  reason, reasoning-closure flag; and the K-1 extra samples (raw, answer
                  section, answer, label, mean/min token logprob)
  {tag}.tok.npz   token-level arrays of the ORIGINAL trace, concatenated with offsets:
                  prompt_ids, out_ids, chosen_lp, ent (renormalized top-k entropy),
                  topk_mean (mean top-k logprob, the DeepConf confidence)

Labels use only the post-reasoning answer section (what the model returned). Questions
are processed in chunks written as shards, so an interrupted run resumes at the first
missing chunk.

  python -m trac.pipeline.generate --model qwen8b --dataset math500 --tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.datasets import load_dataset
from trac.prompts import messages_for
from trac.store import generation_done, write_json_gz


def _flat(comp):
    """(position index, token id, logprob, rank) per flat logprob entry."""
    lp = comp.logprobs
    n = min(len(comp.token_ids), len(lp.start_indices))
    st = np.asarray(lp.start_indices[:n]); en = np.asarray(lp.end_indices[:n])
    pos = np.repeat(np.arange(n), en - st)
    if n and np.array_equal(st[1:], en[:-1]):      # contiguous (the normal case)
        sel = np.arange(st[0], en[-1])
    else:
        sel = np.concatenate([np.arange(a, b) for a, b in zip(st, en)]) if n else np.zeros(0, int)
    tids = np.asarray(lp.token_ids)[sel]
    lps = np.asarray(lp.logprobs, dtype=np.float64)[sel]
    ranks = np.asarray([10 ** 9 if r is None else r for r in lp.ranks])[sel]
    # the sampled token is listed on its own AND inside the top-k: keep one entry per (pos, id)
    _, first = np.unique(pos.astype(np.int64) * (1 << 32) + tids, return_index=True)
    keep = np.sort(first)
    return n, pos[keep], tids[keep], lps[keep], ranks[keep]


def flat_arrays(comp, topk: int):
    """chosen logprob, renormalized top-k entropy, mean top-k logprob per position."""
    n_tok = len(comp.token_ids)
    chosen = np.full(n_tok, np.nan, np.float32)
    ent = np.full(n_tok, np.nan, np.float32)
    tmean = np.full(n_tok, np.nan, np.float32)
    if comp.logprobs is None or n_tok == 0:
        return chosen, ent, tmean
    n, pos, tids, lps, ranks = _flat(comp)
    out = np.asarray(comp.token_ids)
    hit = tids == out[pos]
    chosen[pos[hit]] = lps[hit]
    top = ranks <= topk
    tp, tl = pos[top], lps[top]
    mx = np.full(n, -np.inf); np.maximum.at(mx, tp, tl)
    p = np.exp(tl - mx[tp])
    z = np.bincount(tp, weights=p, minlength=n)
    q = p / z[tp]
    cnt = np.bincount(tp, minlength=n)
    ok = cnt > 0
    ent[:n][ok] = np.bincount(tp, weights=-(q * np.log(q + 1e-12)), minlength=n)[ok]
    tmean[:n][ok] = (np.bincount(tp, weights=tl, minlength=n) / np.maximum(cnt, 1))[ok]
    return chosen, ent, tmean


def sample_record(row, bb, tok, comp):
    raw = tok.decode(comp.token_ids, skip_special_tokens=False)
    final = M.final_for(bb, raw, comp.finish_reason)
    c, _, _ = flat_arrays(comp, 1)
    c = c[np.isfinite(c)]
    return {"raw": raw, "final": final, "answer": A.extract_answer(row, final),
            "correct": A.verify_answer_text(row, final),
            "meanlp": float(c.mean()) if len(c) else None, "minlp": float(c.min()) if len(c) else None,
            "n_tokens": len(comp.token_ids), "finish": comp.finish_reason}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="backbone key in trac.backbones.BACKBONES")
    ap.add_argument("--dataset", required=True, help="dataset tag or mix:ds1,ds2:N")
    ap.add_argument("--tag", required=True, help="cache tag of the outputs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=8, help="samples per question incl. the original trace")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = backbone default budget")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--topk-lp", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk", type=int, default=1024, help="questions per shard")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    args = ap.parse_args()

    out_json = C.GEN / f"{args.tag}.json.gz"
    out_tok = C.GEN / f"{args.tag}.tok.npz"
    if generation_done(args.tag):
        print(f"[gen] exists {out_json}"); return
    shard_dir = C.GEN / f"{args.tag}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.get(args.model)
    tok = M.load_tokenizer(bb)
    rows = load_dataset(args.dataset, limit=args.limit)
    pids = [M.prompt_ids(tok, bb, messages_for(r)) for r in rows]
    prompts = [tok.decode(p, skip_special_tokens=False) for p in pids]
    args.max_tokens = args.max_tokens or bb.max_new
    print(f"[gen] {args.tag}: {len(rows)} q, k={args.k}, model={bb.name}, T={args.temperature}")

    chunks = [list(range(i, min(i + args.chunk, len(rows)))) for i in range(0, len(rows), args.chunk)]
    todo = [ci for ci in range(len(chunks)) if not (shard_dir / f"{ci}.npz").exists()]
    llm = None
    for ci in todo:
        if llm is None:
            llm = M.make_llm(bb, args.gpu_mem, args.seed, max_num_seqs=args.max_num_seqs)
        idx = chunks[ci]
        sp_primary = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, logprobs=args.topk_lp,
                                    flat_logprobs=True, skip_special_tokens=False)
        reqs = [TokensPrompt(prompt_token_ids=pids[i]) for i in idx]
        sps = [sp_primary] * len(idx)
        if args.k > 1:
            sp_extra = SamplingParams(n=args.k - 1, temperature=args.temperature, top_p=args.top_p,
                                      max_tokens=args.max_tokens, logprobs=1,
                                      flat_logprobs=True, skip_special_tokens=False)
            reqs += [TokensPrompt(prompt_token_ids=pids[i]) for i in idx]
            sps += [sp_extra] * len(idx)
        t0 = time.time()
        outs = llm.generate(reqs, sps)
        dt = time.time() - t0

        recs, out_ids, chosen, ent, tmean = [], [], [], [], []
        ntok_total = 0
        for j, i in enumerate(idx):
            row = rows[i]
            comp = outs[j].outputs[0]
            raw = tok.decode(comp.token_ids, skip_special_tokens=False)
            final = M.final_for(bb, raw, comp.finish_reason)
            c, e, m = flat_arrays(comp, args.topk_lp)
            out_ids.append(np.asarray(comp.token_ids, np.int32)); chosen.append(c); ent.append(e); tmean.append(m)
            ntok_total += len(comp.token_ids)
            samples = [sample_record(row, bb, tok, sc) for sc in outs[len(idx) + j].outputs] if args.k > 1 else []
            ntok_total += sum(s["n_tokens"] for s in samples)
            recs.append({
                "id": row["id"], "question": row["question"], "gold": row["gold"],
                "gold_raw": row.get("gold_raw", row["gold"]), "aliases": row.get("aliases"),
                "type": row.get("type", "math"), "n_choices": row.get("n_choices"),
                "category": row.get("category"),
                "prompt": prompts[i], "raw": raw, "final": final, "answer": A.extract_answer(row, final),
                "correct": A.verify_answer_text(row, final),
                "finish": comp.finish_reason, "reason_closed": M.reasoning_end_char(bb, raw) is not None,
                "n_tokens": len(comp.token_ids), "samples": samples,
            })
        np.savez(shard_dir / f"{ci}.npz",
                 prompt_ids=np.concatenate([np.asarray(pids[i], np.int32) for i in idx]),
                 prompt_len=np.asarray([len(pids[i]) for i in idx], np.int64),
                 out_ids=np.concatenate(out_ids), out_len=np.asarray([len(x) for x in out_ids], np.int64),
                 chosen_lp=np.concatenate(chosen), ent=np.concatenate(ent), topk_mean=np.concatenate(tmean))
        with gzip.open(shard_dir / f"{ci}.json.gz", "wt") as f:
            json.dump(recs, f)
        acc = np.mean([r["correct"] for r in recs])
        print(f"[gen] chunk {ci + 1}/{len(chunks)}: {len(idx)} q, {ntok_total} tok in {dt:.0f}s "
              f"({ntok_total / max(dt, 1e-9):.0f} tok/s), acc={acc:.3f}", flush=True)

    # merge shards -> final files
    allrecs, parts = [], {k: [] for k in ["prompt_ids", "prompt_len", "out_ids", "out_len",
                                          "chosen_lp", "ent", "topk_mean"]}
    for ci in range(len(chunks)):
        with gzip.open(shard_dir / f"{ci}.json.gz", "rt") as f:
            allrecs += json.load(f)
        z = np.load(shard_dir / f"{ci}.npz")
        for k in parts:
            parts[k].append(z[k])
    np.savez_compressed(out_tok, **{k: np.concatenate(v) for k, v in parts.items()})
    write_json_gz(out_json, allrecs)
    for p in shard_dir.iterdir():
        p.unlink()
    shard_dir.rmdir()
    print(f"[gen] saved {len(allrecs)} -> {out_json}")


if __name__ == "__main__":
    main()
