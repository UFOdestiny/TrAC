"""Verbalized top-k baseline (one vLLM session per tag): after the end of the reasoning, the frozen
model lists its three most likely final answers with a verbalized confidence for each
(Xiong et al., 2024), from the cached trace prefix. Greedy, at most --max-tokens tokens.

The answer section is prefilled so every backbone follows the format:
  close + "The three most likely final answers, with my confidence in each, are:\\n1. \\boxed{"

Output: EXP_ROOT/topk/{tag}.json.gz, one record per question
  {id, text, items: [{surf, ans, eq_a, conf}]}

  python -m trac.pipeline.topk --gen-tag math500_qwen8b_k8
"""

from __future__ import annotations

import argparse
import re
import time

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.pipeline.contrast import inner
from trac.pipeline.probes import token_chunks
from trac.store import load_generation, write_json_gz

LEAD = "The three most likely final answers, with my confidence in each, are:\n1. \\boxed{"
CONF = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


def parse(text):
    """Items of '1. \\boxed{x} (confidence: y%) 2. \\boxed{...} ...' (the first box is prefilled)."""
    parts = ("\\boxed{" + text).split("\\boxed{")[1:]
    items = []
    for p in parts[:3]:
        surf = inner(p)
        close = p.find("}", len(surf)) if surf else -1  # read the confidence after the box
        m = CONF.search(p[close + 1 :] if close >= 0 else p)
        items.append({"surf": surf, "conf": min(100.0, float(m.group(1))) / 100 if m else None})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--chunk-tokens", type=int, default=600_000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = C.EXP_ROOT / "topk" / f"{a.gen_tag}.json.gz"
    if out.exists():
        print(f"[topk] exists {out}")
        return
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    bb = M.resolve("", a.gen_tag)
    tok = M.load_tokenizer(bb)
    rows = load_generation(a.gen_tag)
    if a.limit:
        rows = rows[: a.limit]
    lead = M.encode(tok, bb.close + LEAD)
    items = []
    for i, r in enumerate(rows):
        body, _ = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        items.append({"i": i, "prefix": r["_pids"] + body + lead})
    fits = lambda ids: len(ids) + a.max_tokens + 8 <= bb.max_model_len
    llm = M.make_llm(bb, a.gpu_mem)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=False)
    res, t0 = {}, time.time()
    for ch in token_chunks(items, a.chunk_tokens):
        live = [it for it in ch if fits(it["prefix"])]
        outs = llm.generate([TokensPrompt(prompt_token_ids=it["prefix"]) for it in live], sp, use_tqdm=False)
        for it, o in zip(live, outs):
            r = rows[it["i"]]
            text = tok.decode(o.outputs[0].token_ids, skip_special_tokens=False)
            its = parse(text)
            for x in its:
                x["ans"] = A.extract_head(r, x["surf"] + "}") if x["surf"] else None
                x["eq_a"] = bool(x["ans"] is not None and A.equiv(r, x["ans"], r["answer"]))
            res[it["i"]] = {"id": r["id"], "text": text, "items": its}
        for it in ch:
            res.setdefault(it["i"], {"id": rows[it["i"]]["id"], "text": "", "items": []})
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_gz(out, [res[i] for i in sorted(res)])
    print(f"[topk] {a.gen_tag}: {len(res)} in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
