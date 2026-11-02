"""Run the whole CPU analysis from the GPU caches, optionally followed by the paper build.

  python -m trac.analysis.run [--jobs 6] [--paper]

Order: features -> in-domain scores -> label-free transfer (views, judge, probe, verifier, few labels,
diagnostics, new backbone) -> paired comparisons, convergence, rivals. Settings run in parallel
processes, each single-threaded; logs go to EXP_ROOT/logs/analysis_*.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from trac import config as C
from trac.benchmarks import DATASET_KEYS, MODEL_KEYS, tag_of

SINGLE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=6, help="parallel processes (default: 6)")
    parser.add_argument("--paper", action="store_true", help="also build the tables and figures in PAPER_DIR")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    tags = sorted(path.name.removesuffix(".json.gz") for path in C.ENDPOINT.glob("*.json.gz"))
    main_tags = [tag_of(d, m) for d in DATASET_KEYS for m in MODEL_KEYS if tag_of(d, m) in tags]
    if not main_tags:
        parser.error(f"no endpoint caches of the main matrix in {C.ENDPOINT}; run the GPU stages first")
    C.LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **SINGLE_THREAD}

    def run(module, *options, log=None):
        command = [sys.executable, "-m", module, *options]
        print("[analysis]", " ".join(command), flush=True)
        with open(C.LOGS / f"analysis_{log or module.rsplit('.', 1)[-1]}.log", "a") as stream:
            subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)

    def per_tag(module, selected, option="--tag"):
        name = module.rsplit(".", 1)[-1]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda tag: run(module, option, tag, log=f"{name}_{tag}"), selected))

    jobs = ("--jobs", str(args.jobs))
    per_tag("trac.analysis.features.build", tags, "--tags")
    per_tag("trac.analysis.features.auxiliary", main_tags, "--tags")
    per_tag("trac.analysis.features.text", main_tags)
    per_tag("trac.analysis.evaluation.scores", main_tags)
    run("trac.analysis.evaluation.scores", "--aggregate")
    per_tag("trac.analysis.transfer.evaluate", main_tags)
    run("trac.analysis.transfer.evaluate", "--aggregate")
    per_tag("trac.analysis.transfer.judge", main_tags)
    run("trac.analysis.transfer.judge", "--aggregate")
    run("trac.analysis.transfer.probe", "--all", str(args.jobs))
    run("trac.analysis.transfer.probe", "--aggregate")
    run("trac.analysis.transfer.verifier")
    per_tag("trac.analysis.transfer.few_labels", main_tags)
    run("trac.analysis.transfer.few_labels", "--aggregate")
    per_tag("trac.analysis.transfer.diagnostics", main_tags)
    run("trac.analysis.transfer.diagnostics", "--aggregate")
    run("trac.analysis.transfer.new_backbone")
    run("trac.analysis.evaluation.pairs", *jobs)
    run("trac.analysis.evaluation.convergence")
    run("trac.analysis.evaluation.rivals", *jobs)
    if args.paper:
        run("trac.paper.build")


if __name__ == "__main__":
    main()
