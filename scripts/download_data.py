"""Download the benchmark files into DATASETS_ROOT (flat mirrors `org_name/`, as read by
trac.datasets). Requires `huggingface_hub`; GPQA is gated, so accept its terms on the Hub
and log in (`huggingface-cli login`) first.

  python scripts/download_data.py [--only gsm8k,math500,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trac.config import DATASETS_ROOT  # noqa: E402

# benchmark tags (trac.datasets) -> (Hub dataset repo, files to fetch)
SOURCES = {
    "gsm8k": [("openai/gsm8k", ["main/test-*.parquet"])],
    "math500": [("HuggingFaceH4/MATH-500", ["test.jsonl"])],
    "minerva": [("math-ai/minervamath", ["test.jsonl"])],
    "olympiad": [("math-ai/olympiadbench", ["test.parquet"])],
    "aime3": [("math-ai/aime24", ["test-*.parquet"]), ("math-ai/aime25", ["test.jsonl"]),
              ("MathArena/aime_2026", ["data/train-*.parquet"])],
    "gpqa_diamond": [("Idavidrein/gpqa", ["gpqa_diamond.csv"])],
    "mmlupro": [("TIGER-Lab/MMLU-Pro", ["data/test-*.parquet"])],
    "triviaqa": [("mandarjoshi/trivia_qa", ["rc.nocontext/validation-*.parquet"])],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of benchmark tags (default: all)")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    tags = args.only.split(",") if args.only else list(SOURCES)
    for tag in tags:
        for repo, patterns in SOURCES[tag]:
            dest = DATASETS_ROOT / repo.replace("/", "_", 1)
            print(f"[data] {tag}: {repo} -> {dest}")
            snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns)


if __name__ == "__main__":
    main()
