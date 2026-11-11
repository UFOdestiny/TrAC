"""Shared job queue over the GPU stages. Run one worker per GPU on the same queue files
(`scripts/run_queue.sh` does this); several comma-separated queue files run in order:

  CUDA_VISIBLE_DEVICES=0 python -m trac.pipeline.queue configs/queues/main.txt &
  CUDA_VISIBLE_DEVICES=1 python -m trac.pipeline.queue configs/queues/main.txt &

Queue lines (`|` separated, `#` comments), one per setting:
  tag | model | dataset | generate options | stages (space separated)

Every setting yields the jobs `{tag}__generate` and `{tag}__{stage}`, each running
`python -m trac.pipeline.<stage> --gen-tag <tag>` after the generation exists (contrast also waits for
the trajectory). A worker claims a job with an atomic mkdir lock under EXP_ROOT/locks and marks it
done or failed; a job whose dependency is still running elsewhere is deferred. Stages skip outputs
that already exist, so remove a failed job's lock directory to retry it. Logs: EXP_ROOT/logs/{job}.log.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time

from trac import config as C

DEPENDS = {"contrast": "trajectory"}


def parse(path):
    """Jobs of a queue file, in order: dicts with job, tag, stage, and the stage's command."""
    jobs = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tag, model, dataset, options, stages = [x.strip() for x in line.split("|")]
        generate = ["--model", model, "--dataset", dataset, "--tag", tag, *shlex.split(options)]
        jobs.append({"job": f"{tag}__generate", "tag": tag, "stage": "generate", "args": generate, "after": None})
        for stage in stages.split():
            after = f"{tag}__{DEPENDS[stage]}" if stage in DEPENDS else f"{tag}__generate"
            jobs.append({"job": f"{tag}__{stage}", "tag": tag, "stage": stage, "args": ["--gen-tag", tag], "after": after})
    return jobs


def state(job):
    lock = C.LOCKS / job
    return (
        "done" if (lock / "done").exists() else "failed" if (lock / "failed").exists() else "running" if lock.exists() else None
    )


def run(job):
    command = [sys.executable, "-m", f"trac.pipeline.{job['stage']}", *job["args"]]
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
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for q in sys.argv[1].split(","):
        work(parse(q))


if __name__ == "__main__":
    main()
