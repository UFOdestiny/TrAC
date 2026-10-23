"""Probes at the COMPLETED reasoning prefix (one vLLM session per tag).

For each stored original trace, with prefix = prompt_ids + reasoning body ids:
  pce       prefix + close + cue_c, greedy short answer (c = 0, 1, 2; cue 0 is PCE,
            cues 1-2 are the cue-robustness variants)
  qonly     question-only control: empty reasoning block + cue 0
  shuffled  shuffled-trace control: another question's reasoning body + cue 0
  tforce    teacher-forced likelihood of the returned answer at prefix + cue 0
  repeat    R stochastic short answers at prefix + cue 0 (re-answer control)
  ptrue     P(True) self-verification on (question, returned answer)
  verbconf  verbalized confidence (0-100) on (question, returned answer); with the
            trace's hedge/verify markers it forms the SelfDoubt baseline

Answer-token logprobs are truncated at the closing brace of the boxed answer, so
likelihood features cover exactly the answer tokens. All prompts are token-exact
continuations of the stored generation (see trac.backbones).

  python -m trac.pipeline.endpoint --gen-tag math500_qwen8b_k8 [--variants ...] [--add]
"""
from __future__ import annotations

import argparse
import random
import re

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import answer_ids, answer_span, chosen_logprobs, cue_ids
from trac.prompts import CUES, PTRUE_QUESTION, VERBAL_CONFIDENCE_QUESTION, messages_for, proposed_answer_messages
from trac.store import load_generation, read_json_gz, write_json_gz

VARIANTS = "pce,qonly,shuffled,tforce,repeat,ptrue,verbconf"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--out-tag", default="", help="defaults to gen-tag (+ probe temperature suffix)")
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--probe-temp", type=float, default=0.0, help="PCE cue-0 decoding temperature")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--variants", default=VARIANTS)
    ap.add_argument("--add", action="store_true", help="run only --variants and merge into an existing output")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_tag = args.out_tag or args.gen_tag + ("" if args.probe_temp == 0 else f"_pt{args.probe_temp:g}")
    out = C.ENDPOINT / f"{out_tag}.json.gz"
    if out.exists() and not args.add:
        print(f"[ep] exists {out}"); return
    variants = set(args.variants.split(","))

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    close_cue = [cue_ids(tok, bb, c) for c in range(len(CUES))]

    bodies, closed = [], []
    for r in rows:
        body, ok = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        bodies.append(body); closed.append(ok)
    rng = random.Random(args.seed)
    perm = list(range(len(rows)))
    while len(rows) > 1 and any(i == p for i, p in enumerate(perm)):
        rng.shuffle(perm)

    reqs, sps, keys = [], [], []
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1,
                            flat_logprobs=True, skip_special_tokens=False)
    pce0 = greedy if args.probe_temp == 0 else SamplingParams(
        n=1, temperature=args.probe_temp, top_p=0.95, max_tokens=args.probe_tokens, logprobs=1,
        flat_logprobs=True, skip_special_tokens=False, seed=args.seed)
    rep = SamplingParams(n=args.repeats, temperature=0.7, top_p=0.95, max_tokens=args.probe_tokens,
                         logprobs=1, flat_logprobs=True, skip_special_tokens=False, seed=args.seed)
    # prompt_logprobs=0 returns the actual-token logprob per prompt row; only the trailing
    # answer rows are read (vLLM 0.31 ignores prompt_logprob_start here)
    tforce = SamplingParams(n=1, temperature=0.0, max_tokens=1, prompt_logprobs=0, flat_logprobs=True)
    bad = 0

    def add(ids, sp, key):
        nonlocal bad
        if len(ids) + args.probe_tokens + 1 > bb.max_model_len:
            bad += 1; return
        reqs.append(TokensPrompt(prompt_token_ids=ids)); sps.append(sp); keys.append(key)

    tf_meta = {}
    for i, r in enumerate(rows):
        prefix = r["_pids"] + bodies[i]
        if "pce" in variants:
            for c in range(len(CUES)):
                add(prefix + close_cue[c], pce0 if c == 0 else greedy, (i, "pce", c))
        if "repeat" in variants and args.repeats > 0:
            add(prefix + close_cue[0], rep, (i, "repeat", 0))
        if "qonly" in variants:
            add(M.direct_ids(tok, bb, messages_for(r)) + M.encode(tok, CUES[0]), greedy, (i, "qonly", 0))
        if "shuffled" in variants:
            add(r["_pids"] + bodies[perm[i]] + close_cue[0], greedy, (i, "shuffled", 0))
        if "tforce" in variants:
            a = A.surface_answer(r)
            if a is not None:
                ans = answer_ids(tok, bb, a)
                tf_meta[i] = (a, ans)
                add(prefix + close_cue[0] + ans, tforce, (i, "tforce", 0))
        if "ptrue" in variants:
            msgs = proposed_answer_messages(r, A.surface_answer(r) or "(no answer)", PTRUE_QUESTION)
            add(M.direct_ids(tok, bb, msgs), None, (i, "ptrue", 0))
        if "verbconf" in variants:
            msgs = proposed_answer_messages(r, A.surface_answer(r) or "(no answer)", VERBAL_CONFIDENCE_QUESTION)
            add(M.direct_ids(tok, bb, msgs) + M.encode(tok, "Confidence: "), greedy, (i, "verbconf", 0))

    # P(True): score the label tokens directly at the first answer position
    tf_ids = {w: M.encode(tok, w)[0] for w in ["True", " True", "true", "False", " False", "false"]}
    pt_ids = sorted(set(tf_ids.values()))
    sp_pt = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=len(pt_ids),
                           logprob_token_ids=pt_ids, flat_logprobs=True)
    sps = [sp_pt if s is None else s for s in sps]
    print(f"[ep] {args.gen_tag}: {len(rows)} q -> {len(reqs)} probe requests "
          f"({bad} skipped for length); reasoning closed {np.mean(closed):.3f}", flush=True)

    llm = M.make_llm(bb, args.gpu_mem, args.seed)
    outs = llm.generate(reqs, sps)

    def comp_rec(r, comp):
        ids = list(comp.token_ids)
        text = tok.decode(ids, skip_special_tokens=False)
        L = answer_span(tok, ids)
        ans = A.extract_head(r, text)
        return {"text": text, "answer": ans, "agree": bool(A.equiv(r, ans, r["answer"])),
                "ntok": L, "tok_lps": chosen_logprobs(comp)[:L]}

    res = [{"id": r["id"], "closed": closed[i], "reason_ntok": len(bodies[i]),
            "answer": r["answer"], "correct": r["correct"], "pce": [None] * len(CUES)}
           for i, r in enumerate(rows)]
    for (i, kind, c), o in zip(keys, outs):
        r = rows[i]
        if kind == "pce":
            res[i]["pce"][c] = comp_rec(r, o.outputs[0])
        elif kind in ("qonly", "shuffled"):
            res[i][kind] = comp_rec(r, o.outputs[0])
            if kind == "shuffled":
                res[i][kind]["src_id"] = rows[perm[i]]["id"]
        elif kind == "repeat":
            res[i]["repeat"] = [{"text": (x := comp_rec(r, comp))["text"], "answer": x["answer"],
                                 "agree": x["agree"], "lp0": x["tok_lps"][0] if x["tok_lps"] else None}
                                for comp in o.outputs]
        elif kind == "tforce":
            a, ans = tf_meta[i]
            pl = o.prompt_logprobs
            scored = [pl[p] for p in range(len(pl)) if pl[p]]
            # scored rows predict the trailing prompt tokens = the "answer}" tokens
            scored = scored[-len(ans):]
            lps = [d[t].logprob if t in d else None for d, t in zip(scored, ans[-len(scored):])]
            res[i]["tforce"] = {"ans": a, "ans_ids": len(ans), "tok_lps": lps}
        elif kind == "verbconf":
            txt = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True)
            m = re.search(r"(\d{1,3}(?:\.\d+)?)", txt)
            res[i]["verbconf"] = {"text": txt, "value": min(100.0, float(m.group(1))) / 100 if m else None}
        elif kind == "ptrue":
            d = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            lt = [d[t].logprob for w, t in tf_ids.items() if w.strip().lower() == "true" and t in d]
            lf = [d[t].logprob for w, t in tf_ids.items() if w.strip().lower() == "false" and t in d]
            pt = np.logaddexp.reduce(lt) if lt else -np.inf
            pf = np.logaddexp.reduce(lf) if lf else -np.inf
            res[i]["ptrue"] = float(1 / (1 + np.exp(pf - pt))) if np.isfinite(pt) or np.isfinite(pf) else 0.5

    if args.add and out.exists():
        prev = {x["id"]: x for x in read_json_gz(out)}
        for x in res:
            p = prev.get(x["id"], {})
            for k, v in x.items():
                if k == "pce" and not any(v):
                    continue
                if k not in ("id", "closed", "reason_ntok", "answer", "correct") or k not in p:
                    p[k] = v
            prev[x["id"]] = p
        res = list(prev.values())
    write_json_gz(out, res)
    agr = [x["pce"][0]["agree"] for x in res if x.get("pce") and x["pce"][0]]
    cor = [x["correct"] for x in res if x.get("pce") and x["pce"][0]]
    if agr:
        a, y = np.array(agr), np.array(cor)
        print(f"[ep] agree={a.mean():.3f} | agree|correct={a[y == 1].mean() if y.sum() else float('nan'):.3f} "
              f"agree|wrong={a[y == 0].mean() if (1 - y).sum() else float('nan'):.3f}")
    print(f"[ep] saved -> {out}")


if __name__ == "__main__":
    main()
