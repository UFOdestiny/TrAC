"""Shared job queue over the GPU stages. Run one worker per GPU on the SAME queue files
(`scripts/run_queue.sh` does this); several comma-separated queues run in order:

  CUDA_VISIBLE_DEVICES=0 python -m trac.pipeline.queue configs/queues/1_baselines.txt,configs/queues/2_trac.txt &
  CUDA_VISIBLE_DEVICES=1 python -m trac.pipeline.queue configs/queues/1_baselines.txt,configs/queues/2_trac.txt &

Queue lines (`|` separated; `#` comments):
  gen_tag | model | dataset | gen_args | ep_out_tag | ep_args | hidden(0/1) [| job_name]

A job claims job_name (default ep_out_tag, else gen_tag) with an atomic mkdir lock under
EXP_ROOT/locks, so one cell can appear in several staged queues (baselines first, TrAC
probes later via `--add`). It then runs
  generate (if the generation cache is missing) -> endpoint -> [hidden_states -> nli] (if 1).
A job whose generation is being produced by the other worker is deferred and retried, so
dependent jobs (e.g. probe-temperature variants) never duplicate work. Every stage is
idempotent; a crashed job is retried after removing its lock directory. The special model
value `latency` runs trac.pipeline.latency with gen_args as its arguments.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time

from trac import config as C
from trac.store import generation_done

PY = sys.executable


def run(cmd, log):
    print(f"[q] $ {' '.join(cmd)}", flush=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[q] FAILED rc={rc} (see {log})", flush=True)
    return rc == 0


def claim(name):
    try:
        os.makedirs(C.LOCKS / name)
        return True
    except FileExistsError:
        return False


def parse(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("|")]
        f += [""] * (8 - len(f))
        jobs.append(dict(gen=f[0], model=f[1], ds=f[2], gen_args=f[3], ep=f[4] or f[0],
                         ep_args=f[5], hidden=f[6] == "1", job=f[7] or f[4] or f[0]))
    return jobs


def stage(name):
    return [PY, "-m", f"trac.pipeline.{name}"]


def do_job(j):
    """True / False when the job finished / failed, None to defer it."""
    log = C.LOGS / f"{j['job']}.log"
    if j["model"] == "latency":
        a = shlex.split(j["gen_args"])
        if not generation_done(a[a.index("--gen-tag") + 1]):
            return None                       # wait for the generation it times
        return run(stage("latency") + a, log)
    if not generation_done(j["gen"]):
        glock = C.LOCKS / ("gen__" + j["gen"])
        if not claim(glock.name):
            # another worker is generating it: defer, unless that generation failed
            return False if (glock / "failed").exists() else None
        ok = run(stage("generate") + ["--model", j["model"], "--dataset", j["ds"], "--tag", j["gen"]]
                 + shlex.split(j["gen_args"]), log)
        if not ok:
            (glock / "failed").write_text("failed\n")
            return False
    if "--add" in j["ep_args"] or not (C.ENDPOINT / f"{j['ep']}.json.gz").exists():
        args = shlex.split(j["ep_args"])
        if j["ep"] != j["gen"] and "--out-tag" not in args:
            args += ["--out-tag", j["ep"]]
        if not run(stage("endpoint") + ["--model", j["model"], "--gen-tag", j["gen"]] + args, log):
            return False
    # main-matrix jobs (last field 1): hidden states + NLI evidence for the white-box and sampling baselines
    if j["hidden"] and not (C.HIDDEN / f"{j['gen']}.npz").exists():
        if not run(stage("hidden_states") + ["--model", j["model"], "--gen-tag", j["gen"]], log):
            return False
    if j["hidden"] and not (C.NLI / f"{j['gen']}.npz").exists():
        if not run(stage("nli") + ["--gen-tag", j["gen"]], log):
            return False
    return True


def work(jobs):
    C.LOCKS.mkdir(parents=True, exist_ok=True); C.LOGS.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    while True:
        pending = 0
        for j in jobs:
            done_flag = C.LOCKS / j["job"] / "done"
            if done_flag.exists() or (C.LOCKS / j["job"] / "failed").exists():
                continue
            if not claim(j["job"]):
                pending += (C.LOCKS / j["job"]).exists() and not done_flag.exists()
                continue
            t0 = time.time()
            print(f"[q gpu{gpu}] START {j['job']}", flush=True)
            res = do_job(j)
            if res is None:                    # deferred: release the claim, retry later
                os.rmdir(C.LOCKS / j["job"]); pending += 1
                print(f"[q gpu{gpu}] DEFER {j['job']} (waiting for gen {j['gen']})", flush=True)
                continue
            (C.LOCKS / j["job"] / ("done" if res else "failed")).write_text(f"gpu{gpu} {time.time() - t0:.0f}s\n")
            print(f"[q gpu{gpu}] {'DONE' if res else 'FAILED'} {j['job']} in {(time.time() - t0) / 60:.1f} min", flush=True)
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
