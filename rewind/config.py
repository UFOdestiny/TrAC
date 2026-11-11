"""Paths and environment, read from the repository's `.env` (see `.env.example`).

Variables already set in the process environment take precedence over `.env`, and `REWIND_ENV_FILE`
selects another file. Relative paths are resolved against the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> dict[str, str]:
    """Load dotenv syntax while preserving values already exported by the caller."""
    from dotenv import dotenv_values

    env = {k: v for k, v in dotenv_values(path, interpolate=False).items() if v is not None}
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def path_from_env(var: str, default: str) -> Path:
    p = Path(os.path.expanduser(os.environ.get(var) or default))
    return p if p.is_absolute() else REPO_ROOT / p


ENV_FILE = Path(os.environ.get("REWIND_ENV_FILE", REPO_ROOT / ".env")).expanduser()
if not ENV_FILE.is_absolute():
    ENV_FILE = REPO_ROOT / ENV_FILE
ENV = load_env(ENV_FILE)
MODELS_ROOT = path_from_env("MODELS_ROOT", "models")  # one folder per model
DATASETS_ROOT = path_from_env("DATASETS_ROOT", "data")  # benchmark mirrors (scripts/download_data.py)
EXP_ROOT = path_from_env("EXP_ROOT", "outputs")  # every cache, score, report, and log
if os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(path_from_env("HF_HOME", ""))

# ---------------------------------------------------------------- artifacts under EXP_ROOT
# GPU stages (rewind.pipeline), one file per cache tag
GEN = EXP_ROOT / "gen"  # {tag}.json.gz + {tag}.tok.npz   generate
ENDPOINT = EXP_ROOT / "endpoint"  # {tag}.json.gz                   endpoint
INTERMEDIATE_ANSWERS = EXP_ROOT / "intermediate_answers"  # {tag}.json.gz  intermediate_answers
RUNNER_UP = EXP_ROOT / "runner_up"  # {tag}.json.gz                   runner_up
# CPU stages (rewind.features, rewind.score)
FEATURES = EXP_ROOT / "features"  # {tag}.npz                       features.build
FEATURE_CACHE = EXP_ROOT / "feature_cache"  # derived blocks, rebuilt when stale
BASE_CACHE = EXP_ROOT / "base_cache"  # content-hashed base-learner predictions
REWIND_SCORES = EXP_ROOT / "rewind_scores"  # {tag}.npz             score
# job queue
LOCKS = EXP_ROOT / "locks"
LOGS = EXP_ROOT / "logs"


def model_path(name: str) -> str:
    """Local folder of a model under MODELS_ROOT."""
    return str(MODELS_ROOT / name)


def main():
    """Print the resolved paths, or shell-safe exports for the Bash launchers."""
    import argparse
    import json
    import shlex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", action="store_true", help="emit quoted Bash export statements")
    args = parser.parse_args()
    paths = {key: str(globals()[key]) for key in ("MODELS_ROOT", "DATASETS_ROOT", "EXP_ROOT")}
    if args.shell:
        for key, value in {**{key: os.environ[key] for key in ENV}, **paths}.items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
