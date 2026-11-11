"""Query at the end of each stored reasoning trace (one vLLM session per tag).

For each stored original trace, with prefix = prompt ids + reasoning body ids:
  end       greedy intermediate answer after the end-of-reasoning marker and the answer cue

Answer-token log-probabilities are truncated at the closing brace of the boxed answer, so likelihood
features cover exactly the answer tokens. All prompts are token-exact continuations of the stored
generation (see rewind.backbones). When the baselines package is present, the same session also asks
its self-evaluation queries (P(True), verbalized confidence; baselines.pipeline.self_evaluation).

  python -m rewind.pipeline.endpoint --gen-tag math500_qwen8b_k8   -> EXP_ROOT/endpoint/{tag}.json.gz
"""

from __future__ import annotations

import argparse

import numpy as np

from rewind import answers as A
from rewind import backbones as M
from rewind import config as C
from rewind.optional import optional_module
from rewind.pipeline.queries import answer_span, chosen_logprobs, cue_ids
from rewind.store import load_generation, write_json_gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--model", default="", help="backbone key (default: inferred from the tag)")
    ap.add_argument("--query-tokens", type=int, default=32)
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
        n=1, temperature=0.0, max_tokens=args.query_tokens, logprobs=1, flat_logprobs=True, skip_special_tokens=False
    )
    self_eval = optional_module("baselines.pipeline.self_evaluation", "the P(True) and verbalized-confidence baselines")
    evaluator = self_eval.SelfEvaluation(tok, bb, greedy) if self_eval else None

    reqs, sps, keys, closed, lengths, bad = [], [], [], [], [], 0
    for i, r in enumerate(rows):
        body, ok = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        closed.append(ok)
        lengths.append(len(body))
        for kind, ids, sp in [("end", r["_pids"] + body + close_cue, greedy)] + (evaluator.requests(r) if evaluator else []):
            if len(ids) + args.query_tokens + 1 > bb.max_model_len:
                bad += 1
                continue
            reqs.append(TokensPrompt(prompt_token_ids=ids))
            sps.append(sp)
            keys.append((i, kind))
    print(
        f"[ep] {args.gen_tag}: {len(rows)} q -> {len(reqs)} requests "
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
        else:
            key, value = evaluator.parse(kind, o)
            res[i][key] = value

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
