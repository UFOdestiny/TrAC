"""Paths and environment, read from the repository's `.env` (see `.env.example`).

Variables already set in the process environment take precedence over `.env`.
Relative paths are resolved against the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = REPO_ROOT / ".env") -> Dict[str, str]:
    """Parse KEY=VALUE lines (blank lines, `#` comments and inline ` #` comments ignored)
    and export them with `os.environ.setdefault`."""
    env: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip().strip('"').strip("'")
            env[k.strip()] = v
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def _path(var: str, default: str) -> Path:
    p = Path(os.path.expanduser(os.environ.get(var) or default))
    return p if p.is_absolute() else REPO_ROOT / p


ENV = load_env()
MODELS_ROOT = _path("MODELS_ROOT", "models")          # one folder per backbone / NLI model
DATASETS_ROOT = _path("DATASETS_ROOT", "data")        # flat dataset mirrors (scripts/download_data.py)
EXP_ROOT = _path("EXP_ROOT", "outputs")               # every cache, score, and log
PAPER_DIR = _path("PAPER_DIR", "latex")               # receives table/*.tex and figures/*.pdf
if os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_path("HF_HOME", ""))
EXP_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- artifact layout under EXP_ROOT
GEN = EXP_ROOT / "gen"                        # {tag}.json.gz + {tag}.tok.npz       pipeline.generate
ENDPOINT = EXP_ROOT / "endpoint"              # {tag}.json.gz                       pipeline.endpoint
TRAJECTORY = EXP_ROOT / "trajectory"          # {tag}.json.gz                       pipeline.trajectory
SAMPLE_TRAJECTORY = EXP_ROOT / "sample_trajectory"  # {tag}.json.gz                 pipeline.sample_trajectory
PREFIX_CONSISTENCY = EXP_ROOT / "prefix_consistency"  # {tag}.json.gz               pipeline.prefix_consistency
HIDDEN = EXP_ROOT / "hidden"                  # {tag}.npz                           pipeline.hidden_states
NLI = EXP_ROOT / "nli"                        # {tag}.npz                           pipeline.nli
LATENCY = EXP_ROOT / "latency"                # {endpoint,trajectory}_{name}.json   pipeline.latency
FEATURES = EXP_ROOT / "features"              # {tag}.npz                           analysis.features
FEATURE_CACHE = EXP_ROOT / "feature_cache"    # derived blocks, rebuilt when stale  analysis.trajectory_features
AUXILIARY = EXP_ROOT / "auxiliary"            # {tag}.npz                           analysis.auxiliary
SCORES = EXP_ROOT / "scores"                  # {tag}.npz out-of-fold scores        analysis.scores
WITHIN_FOLD = EXP_ROOT / "within_fold"        # {tag}.json                          analysis.within_fold
BASE_CACHE = EXP_ROOT / "base_cache"          # content-hashed base-learner predictions
LOCKS = EXP_ROOT / "locks"                    # job queue state                     pipeline.queue
LOGS = EXP_ROOT / "logs"


def model_path(name: str) -> str:
    """Local folder of a model under MODELS_ROOT."""
    return str(MODELS_ROOT / name)
