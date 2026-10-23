"""Registry of evaluated methods: TrAC, its ablations, and every baseline.

Learned methods (METHODS) are scored out of fold by trac.analysis.scores; raw baselines
(raw_baselines) are used as scores directly. Budget classes:
  single-gray   one trace, token log-probabilities (TrAC's class)
  single-white  one trace, hidden states
  k2 / k8       2 / 8 sampled traces
  ablation      variants reported in the ablation tables
Raw-only classes: trac0 (training-free TrAC-0), single-gray-pc (prefix consistency).

Feature blocks come from trac.analysis.features and trac.analysis.cells:
  PCE  completed-prefix re-elicitation (agreement, likelihood, confidence, cue paraphrases,
       stochastic re-answers, teacher-forced likelihood, contrast) + the trajectory (TRAJ)
  TUP  passive trace profile + answer-region confidence
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from trac.analysis.heads import logit

Blocks = Dict[str, np.ndarray]


# ----------------------------------------------------------------------------- feature views
def endpoint_f(B):
    """Completed-prefix PCE features (no trajectory)."""
    return np.hstack([B["AGREE"], B["FLL"], B["FCONF"], B["CUES"], B["REPEAT"], B["TFORCE"], B["CONTRAST"]])


def tup_f(B):
    return np.hstack([B["TRACE"], B["ANS"]])


def pce_f(B):
    return np.hstack([endpoint_f(B), B["TRAJ"]])


def trac_f(B):
    return np.hstack([endpoint_f(B), tup_f(B), B["TRAJ"]])


def with_trajectory(B, block):
    """TrAC features with the trajectory block replaced by an ablated version."""
    return np.hstack([endpoint_f(B), tup_f(B), B[block]])


def prior_f(B):
    """The six prior single-trace signals: mean log-prob, self-certainty, DeepConf-bottom,
    P(True), and the SelfDoubt inputs."""
    return np.hstack([B["BASE"], B["PT"], B["SD"]])


def anchors8(B):
    """K = 8 consensus anchors: logit primary / modal vote share, answer entropy, CISC, SE (lik.)."""
    sc, sp = B["SC"], B["SAMP"]
    return np.column_stack([logit(np.clip(sc[:, 0], 0.01, 0.99)), logit(np.clip(sc[:, 1], 0.01, 0.99)),
                            sc[:, 2], logit(np.clip(sp[:, 8], 0.01, 0.99)), sp[:, 2]])


def hs_bases(H):
    """Hidden-state base learners: answer end and PCE cue, layer 0.75L."""
    return [(H[:, 1, 1], "lr_hs"), (H[:, 3, 1], "lr_hs")]


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Method:
    budget: str
    build: Callable[[Blocks, Optional[np.ndarray]], object]   # (B, H) -> X, stack spec, or None
    head: str                                                 # stack | lr | lr_hs | gb | platt
    ours: bool = False


def stack(view, *, needs=(), hidden=False, anchors=None):
    """Final TrAC head on a feature view: cross-fitted LR + GB stack, optionally with the
    hidden-state bases and raw anchors. None when a required block is missing."""
    def build(B, H):
        if any(k not in B for k in needs) or (hidden and H is None):
            return None
        X = view(B)
        return {"bases": [(X, "lr"), (X, "gb")] + (hs_bases(H) if hidden else []),
                "anchors": anchors(B) if anchors else None}
    return build


def sc2_f(B):
    return logit(np.clip(B["SC2"], 0.01, 0.99))


def tsc_anchors(B):
    return np.hstack([anchors8(B), logit(np.clip(B["TSC"][:, :1], 0.01, 0.99))])


METHODS: Dict[str, Method] = {
    # --- TrAC (ours) ---
    "TrAC": Method("single-gray", stack(trac_f), "stack", ours=True),
    "TrAC + HS": Method("single-white", stack(trac_f, hidden=True), "stack", ours=True),
    "TrAC + CF@8": Method("k8", stack(trac_f, anchors=anchors8), "stack", ours=True),
    "TrAC + SC@2": Method("k2", stack(lambda B: np.hstack([trac_f(B), sc2_f(B)]), needs=("SC2",)), "stack", ours=True),
    "TrAC + PC": Method("single-gray", stack(lambda B: np.hstack([trac_f(B), B["PC"]]), needs=("PC",)), "stack", ours=True),
    "TrAC + CF@8 + TSC@8": Method("k8", stack(trac_f, needs=("TSC",), anchors=tsc_anchors), "stack", ours=True),
    # --- components ---
    "TUP": Method("single-gray", stack(tup_f), "stack"),
    "PCE": Method("single-gray", stack(pce_f), "stack"),
    "PCE (completed prefix only)": Method("single-gray", stack(endpoint_f), "stack"),
    "Trajectory only": Method("single-gray", stack(lambda B: B["TRAJ"]), "stack"),
    "TrAC without trajectory": Method("single-gray", stack(lambda B: np.hstack(
        [B["AGREE"], B["FLL"], B["FCONF"], B["TRACE"], B["CUES"], B["REPEAT"], B["TFORCE"], B["ANS"], B["CONTRAST"]])),
        "stack"),
    # --- matched-supervision comparators ---
    "Supervised prior signals": Method("single-gray", stack(prior_f), "stack"),
    "Text position": Method("single-gray", stack(lambda B: B["TEXTPOS"], needs=("TEXTPOS",)), "stack"),
    "HS-probe": Method("single-white", lambda B, H: None if H is None else H[:, 1, 1], "lr_hs"),
    "Liu et al. (2024)": Method("single-white", lambda B, H: None if H is None else
                                np.hstack([H[:, 1, 0], H[:, 1, 2], B["BASE"], B["TRACE"][:, 16:]]), "lr_hs"),
    "TUP + HS": Method("single-white", stack(tup_f, hidden=True), "stack"),
    "TUP + SC@2": Method("k2", stack(lambda B: np.hstack([tup_f(B), sc2_f(B)]), needs=("SC2",)), "stack"),
    "TUP + CF@8": Method("k8", stack(tup_f, anchors=anchors8), "stack"),
    "Supervised Consensus@8": Method("k8", lambda B, H: np.hstack([B["SC"][:, 1:3], B["TRACE"], B["FLL"][:, :1]]), "lr"),
    "SC@8 (Platt)": Method("k8", lambda B, H: logit(np.clip(B["SC"][:, :1], 0.01, 0.99)), "lr"),
    "SC@8 (Platt, fixed C)": Method("k8", lambda B, H: logit(np.clip(B["SC"][:, :1], 0.01, 0.99)), "platt"),
    # --- ablations ---
    "Abl: logistic head": Method("ablation", lambda B, H: trac_f(B), "lr"),
    "Abl: completed-trace probe": Method("ablation", stack(lambda B: np.hstack([B["AGREE"], B["FLL"], B["FCONF"]])), "stack"),
    "Abl: question-only probe": Method("ablation", stack(lambda B: B["QONLY"]), "stack"),
    "Abl: shuffled-trace probe": Method("ablation", stack(lambda B: B["SHUF"]), "stack"),
    "Abl: teacher-forced support": Method("ablation", stack(lambda B: B["TFORCE"]), "stack"),
    "Abl: agreement only": Method("ablation", stack(lambda B: B["AGREE"]), "stack"),
    "Abl: likelihood only": Method("ablation", stack(lambda B: np.hstack([B["FLL"], B["FCONF"]])), "stack"),
    "Abl: + re-elicitation distribution": Method("ablation", stack(lambda B: np.hstack(
        [B["AGREE"], B["FLL"], B["FCONF"], B["CUES"], B["REPEAT"], B["TFORCE"]])), "stack"),
    "Abl: P(True) + PCE": Method("ablation", stack(lambda B: np.hstack([endpoint_f(B), B["PT"]])), "stack"),
    "Abl: trajectory, greedy only": Method("ablation", stack(lambda B: with_trajectory(B, "TRAJ_G")), "stack"),
    "Abl: trajectory, 8 checkpoints": Method("ablation", stack(lambda B: with_trajectory(B, "TRAJ_8")), "stack"),
    "Abl: trajectory, 4 checkpoints": Method("ablation", stack(lambda B: with_trajectory(B, "TRAJ_4")), "stack"),
    "Abl: TUP + text position": Method("ablation", stack(lambda B: np.hstack([tup_f(B), B["TEXTPOS"]]),
                                                         needs=("TEXTPOS",)), "stack"),
    "Abl: TrAC + text position": Method("ablation", stack(lambda B: np.hstack([trac_f(B), B["TEXTPOS"]]),
                                                          needs=("TEXTPOS",)), "stack"),
}


def raw_baselines(B: Blocks) -> Dict[str, Dict[str, np.ndarray]]:
    """Unsupervised scores (higher = more confident), grouped by budget class."""
    sd, sp = B["SD"], B["SAMP"]
    V = sd[:, 1]
    return {
        "single-gray-pc": ({"Prefix consistency": np.nan_to_num(B["PC"][:, 0], nan=0.5)} if "PC" in B else {}),
        "trac0": {k: B[c][:, 0] for k, c in [("TrAC-0", "TRAC0"), ("TrAC-0 (greedy only)", "TRAC0_G"),
                                              ("TrAC-0 (8 checkpoints)", "TRAC0_8")] if c in B},
        "single-gray": {"Mean log-prob": B["BASE"][:, 0], "Self-certainty": B["BASE"][:, 1],
                        "DeepConf-bottom": B["BASE"][:, 2], "P(True)": B["PT"][:, 0],
                        "Verbalized confidence": V,
                        "SelfDoubt": z(sd[:, 0]) + (z(V) if np.isfinite(V).all() else 0.0)},
        "k8": {**({"Trajectory-weighted SC@8": B["TSC"][:, 0]} if "TSC" in B else {}),
               "SC@8 (primary)": B["SC"][:, 0], "SC@8 (modal)": B["SC"][:, 1], "CISC@8": sp[:, 8],
               "Semantic entropy@8": sp[:, 1], "Semantic entropy@8 (lik.)": sp[:, 2],
               "SelfCheckGPT@8 (NLI)": sp[:, 0], "EigV@8": sp[:, 4], "Deg@8": sp[:, 6], "KLE@8": sp[:, 7]},
    }

