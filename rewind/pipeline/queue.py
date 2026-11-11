"""Shared job queue over the GPU stages, built from the benchmark and backbone registries
(rewind.benchmarks). Run one worker per GPU on the same matrices (`scripts/run_queue.sh` does this);
several comma-separated matrices run in order:

  CUDA_VISIBLE_DEVICES=0 python -m rewind.pipeline.queue main &
  CUDA_VISIBLE_DEVICES=1 python -m rewind.pipeline.queue main &
  python -m rewind.pipeline.queue main,new_backbone --list     print the jobs without running them

Matrices:
  main           every benchmark x the four main backbones, K = 8 traces per question
  new_backbone   every benchmark x the held-out backbone, one trace per question

Every setting yields the jobs `{tag}__generate` and `{tag}__{stage}` for the REWIND stages (endpoint,
intermediate_answers, runner_up), followed by the stages that optional companion packages register
through `queue_stages(matrix, dataset, model)` in their `pipeline` package; when such a package is
absent, its stages are skipped. A stage job runs `python -m <package>.pipeline.<stage> --gen-tag <tag>`
after the generation exists (runner_up also waits for intermediate_answers). A worker claims a job with
an atomic mkdir lock under EXP_ROOT/locks and marks it done or failed; a job whose dependency is still
running elsewhere is deferred. Stages skip outputs that already exist, so remove a failed job's lock
directory to retry it. Logs: EXP_ROOT/logs/{job}.log.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time

from rewind import config as C
from rewind.benchmarks import DATASET_KEYS, MODEL_KEYS, NEW_BACKBONE, K, tag_of
from rewind.optional import optional_module

MATRICES = {
    "main": ([(d, m) for d in DATASET_KEYS for m in MODEL_KEYS], K),
    "new_backbone": ([(d, NEW_BACKBONE) for d in DATASET_KEYS], 1),
}
REWIND_STAGES = ["endpoint", "intermediate_answers", "runner_up"]
COMPANIONS = ["analysis.pipeline", "baselines.pipeline"]  # optional, not part of the release
DEPENDS = {"runner_up": "intermediate_answers"}


def stage_module(stage):
    """Module that runs a stage: rewind.pipeline.<stage>, else <companion>.<stage>; None if absent."""
    for package in ["rewind.pipeline", *COMPANIONS]:
        try:
            if importlib.util.find_spec(f"{package}.{stage}") is not None:
                return f"{package}.{stage}"
        except ModuleNotFoundError:  # the companion package itself is absent
            continue
    return None


def jobs(matrix):
    """Jobs of a matrix, in order: dicts with job, tag, stage, module, and the stage's arguments."""
    settings, k = MATRICES[matrix]
    companions = [m for m in (optional_module(p, f"its GPU stages in the {matrix} matrix") for p in COMPANIONS) if m]
    out = []
    for dataset, model in settings:
        tag = tag_of(dataset, model, k)
        generate = ["--model", model, "--dataset", dataset, "--tag", tag, "--k", str(k)]
        out.append({"job": f"{tag}__generate", "tag": tag, "stage": "generate", "args": generate, "after": None})
        stages = REWIND_STAGES + [s for package in companions for s in package.queue_stages(matrix, dataset, model)]
        for stage in stages:
            after = f"{tag}__{DEPENDS[stage]}" if stage in DEPENDS else f"{tag}__generate"
            out.append({"job": f"{tag}__{stage}", "tag": tag, "stage": stage, "args": ["--gen-tag", tag], "after": after})
    for j in out:
        j["module"] = stage_module(j["stage"])
    return out


def state(job):
    lock = C.LOCKS / job
    return (
        "done" if (lock / "done").exists() else "failed" if (lock / "failed").exists() else "running" if lock.exists() else None
    )


def run(job):
    command = [sys.executable, "-m", job["module"], *job["args"]]
    log = C.LOGS / f"{job['job']}.log"
    print(f"[q] $ {' '.join(command)}", flush=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(command)}\n")
        f.flush()
        return subprocess.call(command, stdout=f, stderr=subprocess.STDOUT) == 0


def work(jobs):
    C.LOCKS.mkdir(parents=True, exist_ok=True)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    while True:
        pending = 0
        for j in jobs:
            if state(j["job"]) in ("done", "failed"):
                continue
            before = state(j["after"]) if j["after"] else "done"
            if before == "failed":  # the dependency failed: so does this job
                (C.LOCKS / j["job"]).mkdir(exist_ok=True)
                (C.LOCKS / j["job"] / "failed").write_text(f"dependency {j['after']} failed\n")
                continue
            if before != "done":
                pending += 1
                continue
            try:
                os.makedirs(C.LOCKS / j["job"])
            except FileExistsError:  # claimed by another worker
                pending += 1
                continue
            t0 = time.time()
            print(f"[q gpu{gpu}] START {j['job']}", flush=True)
            ok = run(j)
            (C.LOCKS / j["job"] / ("done" if ok else "failed")).write_text(f"gpu{gpu} {time.time() - t0:.0f}s\n")
            print(f"[q gpu{gpu}] {'DONE' if ok else 'FAILED'} {j['job']} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not pending:
            break
        time.sleep(60)
    print(f"[q gpu{gpu}] queue finished", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrices", nargs="?", default="", help=f"comma-separated matrices: {', '.join(MATRICES)}")
    ap.add_argument("--list", action="store_true", help="print the jobs and their state, run nothing")
    ap.add_argument("--module", metavar="STAGE", help="print the module that runs STAGE and exit")
    args = ap.parse_args()
    if args.module:
        module = stage_module(args.module)
        if module is None:
            sys.exit(f"stage {args.module!r} not found in rewind.pipeline or the companion packages")
        print(module)
        return
    names = [m for m in args.matrices.split(",") if m]
    unknown = [m for m in names if m not in MATRICES]
    if not names or unknown:
        ap.error(f"pass one or more of: {', '.join(MATRICES)}")
    for name in names:
        if args.list:
            for j in jobs(name):
                print(f"{j['job']:55s} {j['module']:50s} {state(j['job']) or '-'}")
        else:
            work(jobs(name))


if __name__ == "__main__":
    main()
