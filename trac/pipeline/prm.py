"""Process reward model baseline for the mathematical benchmarks (one GPU): Qwen2.5-Math-PRM-7B scores
every step of the response a user sees (the answer section of thinking backbones, the whole
response of instruct backbones; trac.pipeline.reward.response_text), with steps split at blank
lines (the last 40 steps when longer). Each response gets the minimum and the last step reward.

Writes EXP_ROOT/prm/{tag}.npz with ids, prm_min, prm_last.

  python -m trac.pipeline.prm --gen-tag math500_qwen8b_k8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.config import model_path
from trac.pipeline.reward import response_text
from trac.store import read_json_gz, save_npz_atomic

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
MAX_STEPS, MAX_TOKENS = 40, 4096  # the PRM's context length


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--prm", default="Qwen2.5-Math-PRM-7B")
    a = ap.parse_args()
    out = C.EXP_ROOT / "prm" / f"{a.gen_tag}.npz"
    if out.exists():
        print(f"[prm] exists {out}")
        return
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    bb = M.from_tag(a.gen_tag)
    tok = AutoTokenizer.from_pretrained(model_path(a.prm), trust_remote_code=True)
    llm = LLM(
        model=model_path(a.prm),
        runner="pooling",
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=MAX_TOKENS,
    )
    rows = read_json_gz(C.GEN / f"{a.gen_tag}.json.gz")
    prompts = []
    for r in rows:
        steps = [s.strip() for s in response_text(bb, r).split("\n\n") if s.strip()][-MAX_STEPS:] or ["(empty)"]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["question"][:3000]},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        if len(ids) > MAX_TOKENS - 8:  # keep the question and the last steps
            head = ids[:400]
            tail = ids[-(MAX_TOKENS - 8 - 400) :]
            ids = head + tail
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    t0 = time.time()
    outs = llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)
    mn, last = np.full(len(rows), np.nan, np.float32), np.full(len(rows), np.nan, np.float32)
    for i, o in enumerate(outs):
        d = o.outputs.data
        if d is not None and len(d):
            p = d[:, 1].float().numpy()
            mn[i], last[i] = float(p.min()), float(p[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(out, ids=np.array([str(r["id"]) for r in rows]), prm_min=mn, prm_last=last)
    print(f"[prm] {a.gen_tag}: {len(rows)} in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
