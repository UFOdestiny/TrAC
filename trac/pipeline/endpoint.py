"""Probes at the end of each stored reasoning trace (one vLLM session per tag).

For each stored original trace, with prefix = prompt ids + reasoning body ids:
  end       greedy intermediate answer after the end-of-reasoning marker and the answer cue
  ptrue     P(True) self-verification on (question, returned answer)
  verbconf  verbalized confidence (0-100) on (question, returned answer); with the trace's
            hedge/verify markers it forms the SelfDoubt baseline

Answer-token log-probabilities are truncated at the closing brace of the boxed answer, so likelihood
features cover exactly the answer tokens. All prompts are token-exact continuations of the stored
generation (see trac.backbones).

  python -m trac.pipeline.endpoint --gen-tag math500_qwen8b_k8   -> EXP_ROOT/endpoint/{tag}.json.gz
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.probes import answer_span, chosen_logprobs, cue_ids
from trac.prompts import PTRUE_QUESTION, VERBAL_CONFIDENCE_QUESTION, proposed_answer_messages
from trac.store import load_generation, write_json_gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = C.ENDPOINT / f"{args.gen_tag}.json.gz"
    if out.exists():
        print(f"[ep] exists {out}")
        return

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve(args.model, args.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(args.gen_tag)
    close_cue = cue_ids(tok, bb)
    greedy = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1, flat_logprobs=True, skip_special_tokens=False
    )
    # P(True): score the label tokens directly at the first answer position
    tf_ids = {w: M.encode(tok, w)[0] for w in ["True", " True", "true", "False", " False", "false"]}
    pt_ids = sorted(set(tf_ids.values()))
    ptrue = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=len(pt_ids), logprob_token_ids=pt_ids, flat_logprobs=True)

    reqs, sps, keys, closed, lengths, bad = [], [], [], [], [], 0
    for i, r in enumerate(rows):
        body, ok = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        closed.append(ok)
        lengths.append(len(body))
        answer = A.surface_answer(r) or "(no answer)"
        for kind, ids, sp in [
            ("end", r["_pids"] + body + close_cue, greedy),
            ("ptrue", M.direct_ids(tok, bb, proposed_answer_messages(r, answer, PTRUE_QUESTION)), ptrue),
            (
                "verbconf",
                M.direct_ids(tok, bb, proposed_answer_messages(r, answer, VERBAL_CONFIDENCE_QUESTION))
                + M.encode(tok, "Confidence: "),
                greedy,
            ),
        ]:
            if len(ids) + args.probe_tokens + 1 > bb.max_model_len:
                bad += 1
                continue
            reqs.append(TokensPrompt(prompt_token_ids=ids))
            sps.append(sp)
            keys.append((i, kind))
    print(
        f"[ep] {args.gen_tag}: {len(rows)} q -> {len(reqs)} probe requests "
        f"({bad} skipped for length); reasoning closed {np.mean(closed):.3f}",
        flush=True,
    )

    llm = M.make_llm(bb, args.gpu_mem, args.seed)
    outs = llm.generate(reqs, sps)

    def comp_rec(r, comp):
        ids = list(comp.token_ids)
        text = tok.decode(ids, skip_special_tokens=False)
        L = answer_span(tok, ids)
        ans = A.extract_head(r, text)
        return {
            "text": text,
            "answer": ans,
            "agree": bool(A.equiv(r, ans, r["answer"])),
            "ntok": L,
            "tok_lps": chosen_logprobs(comp)[:L],
        }

    res = [
        {
            "id": r["id"],
            "closed": closed[i],
            "reason_ntok": lengths[i],
            "answer": r["answer"],
            "correct": r["correct"],
            "end": None,
        }
        for i, r in enumerate(rows)
    ]
    for (i, kind), o in zip(keys, outs):
        r = rows[i]
        if kind == "end":
            res[i]["end"] = comp_rec(r, o.outputs[0])
        elif kind == "verbconf":
            txt = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True)
            m = re.search(r"(\d{1,3}(?:\.\d+)?)", txt)
            res[i]["verbconf"] = {"text": txt, "value": min(100.0, float(m.group(1))) / 100 if m else None}
        else:
            d = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            lt = [d[t].logprob for w, t in tf_ids.items() if w.strip().lower() == "true" and t in d]
            lf = [d[t].logprob for w, t in tf_ids.items() if w.strip().lower() == "false" and t in d]
            pt = np.logaddexp.reduce(lt) if lt else -np.inf
            pf = np.logaddexp.reduce(lf) if lf else -np.inf
            res[i]["ptrue"] = float(1 / (1 + np.exp(pf - pt))) if np.isfinite(pt) or np.isfinite(pf) else 0.5

    write_json_gz(out, res)
    agr = [x["end"]["agree"] for x in res if x["end"]]
    cor = [x["correct"] for x in res if x["end"]]
    if agr:
        a, y = np.array(agr), np.array(cor)
        print(
            f"[ep] agree={a.mean():.3f} | agree|correct={a[y == 1].mean() if y.sum() else float('nan'):.3f} "
            f"agree|wrong={a[y == 0].mean() if (1 - y).sum() else float('nan'):.3f}"
        )
    print(f"[ep] saved -> {out}")


if __name__ == "__main__":
    main()
