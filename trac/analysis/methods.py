"""Feature views and the registry of evaluated methods.

Feature blocks come from trac.analysis.cells.load_cell:
  token-probability features (phi_P)
    TRACE   binned token log-probabilities and entropies of the original generation, with trends
    ANS     confidence over the answer section and the boxed answer
    META    output length, reasoning closed, truncated
  intermediate-answer features (phi_A)
    TRAJ_4  intermediate answers at four checkpoints and the end of the trace (trajectory)
    TFT     likelihood of the final answer a at the end of the trace
    CON     rival answers: share and margin of a against the model's own alternatives

Learned methods (METHODS) are scored out of fold by trac.analysis.evaluation.scores; training-free
scores (raw_baselines) are used as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from trac.analysis.heads import logit

Blocks = Dict[str, np.ndarray]

# columns of TRAJ_4 computed from the likelihood of a (level and late-half rise) rather than decoded answers
LIKELIHOOD_COLUMNS = [49, 50]
DECODED_COLUMNS = [j for j in range(52) if j not in LIKELIHOOD_COLUMNS]


# ----------------------------------------------------------------------------- feature views
def length_f(B):
    """Log output length, reasoning closed, truncated."""
    return np.hstack([np.log1p(B["META"][:, :1]), B["META"][:, 1:]])


def token_f(B):
    """Token-probability features phi_P."""
    return np.hstack([B["TRACE"], B["ANS"], length_f(B)])


def answer_f(B):
    """Intermediate-answer features phi_A: trajectory, likelihood of a, and rival answers."""
    return np.hstack([B["TRAJ_4"], B["TFT"], B["CON"]])


def trac_f(B):
    return np.hstack([token_f(B), answer_f(B)])


def minimal_f(B):
    """Minimal configuration: greedy answers at four checkpoints, likelihood of a, rival at the end."""
    return np.hstack([token_f(B), B["TRAJ_G4"], B["TFT"], B["CON_END"]])


def early_exit_f(B):
    """Early-exit signals from the 20-checkpoint intermediate answers: answer convergence point,
    Certaindex-style late agreement, and DEER-style end confidence."""
    T = B["TRAJ"]
    return np.column_stack([T[:, 44], np.nanmean(T[:, 16:21], axis=1), B["FLL"][:, 0]])


def likelihood_f(B):
    """Likelihood of a along the trace and at its end (no decoded answers)."""
    return np.hstack([B["TFT"], B["TRAJ_4"][:, LIKELIHOOD_COLUMNS]])


def decoded_f(B):
    """Trajectory features computed from decoded intermediate answers only."""
    return B["TRAJ_4"][:, DECODED_COLUMNS]


def prior_f(B):
    """Prior single-trace signals: mean log-probability, Self-Certainty, DeepConf-LGC, P(True),
    and the SelfDoubt inputs."""
    return np.hstack([B["BASE"], B["PT"], B["SD"]])


def consensus_f(B):
    """Eight-sample consensus features: logit vote share of a and of the modal answer, answer
    entropy, CISC, and likelihood-weighted semantic entropy."""
    sc, sp = B["SC"], B["SAMP"]
    return np.column_stack(
        [
            logit(np.clip(sc[:, 0], 0.01, 0.99)),
            logit(np.clip(sc[:, 1], 0.01, 0.99)),
            sc[:, 2],
            logit(np.clip(sp[:, 8], 0.01, 0.99)),
            sp[:, 2],
        ]
    )


def probe_bases(H):
    """Linear hidden-state probes (layer 0.75L) at the answer end and at the answer cue."""
    return [(H[:, 1, 1], "lr_hs"), (H[:, 3, 1], "lr_hs")]


def with_(*views):
    return lambda B: np.hstack([v(B) for v in views])


def block(name, columns=None):
    return lambda B: B[name] if columns is None else B[name][:, columns]


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Method:
    budget: str  # grey-box | white-box | judge | 8 samples | ablation
    build: Callable[[Blocks, Optional[np.ndarray]], object]  # (B, H) -> X, stack spec, or None
    head: str  # stack | lr | lr_hs
    ours: bool = False


def stack(view, *, needs=(), probes=False, anchors=None):
    """The TrAC classifier on a feature view: cross-fitted LR + GB stack, optionally with the
    hidden-state probes and raw anchor features. None when a required block is missing."""

    def build(B, H):
        if any(k not in B for k in needs) or (probes and H is None):
            return None
        X = view(B)
        return {"bases": [(X, "lr"), (X, "gb")] + (probe_bases(H) if probes else []), "anchors": anchors(B) if anchors else None}

    return build


METHODS: Dict[str, Method] = {
    # --- TrAC ---
    "TrAC": Method("grey-box", stack(trac_f), "stack", ours=True),
    "TrAC (minimal)": Method("grey-box", stack(minimal_f, needs=("CON_END",)), "stack", ours=True),
    "TrAC + probes": Method("white-box", stack(trac_f, probes=True), "stack", ours=True),
    "TrAC@8": Method("8 samples", stack(trac_f, anchors=consensus_f), "stack", ours=True),
    "TrAC + RM": Method("judge", stack(with_(trac_f, block("RM")), needs=("RM",)), "stack", ours=True),
    "TrAC + PRM": Method("judge", stack(with_(trac_f, block("PRM")), needs=("PRM",)), "stack", ours=True),
    # --- feature groups with the same classifier ---
    "Token probs.": Method("grey-box", stack(token_f), "stack"),
    "Intermediate answers": Method("grey-box", stack(answer_f), "stack"),
    "Supervised prior signals": Method("grey-box", stack(prior_f), "stack"),
    "Token probs. + early-exit signals": Method("grey-box", stack(with_(token_f, early_exit_f)), "stack"),
    "Token probs. + likelihood of a": Method("grey-box", stack(with_(token_f, likelihood_f)), "stack"),
    "Token probs. + intermediate answers": Method("grey-box", stack(with_(token_f, decoded_f)), "stack"),
    "Token probs. + verbalized top-k": Method("grey-box", stack(with_(token_f, block("TOPK")), needs=("TOPK",)), "stack"),
    "Token probs. + rival answers": Method("grey-box", stack(with_(token_f, block("CON"))), "stack"),
    "Token probs. + PRM": Method("judge", stack(with_(token_f, block("PRM")), needs=("PRM",)), "stack"),
    "Token probs. + RM": Method("judge", stack(with_(token_f, block("RM")), needs=("RM",)), "stack"),
    "Token probs. + probes": Method("white-box", stack(token_f, probes=True), "stack"),
    "Token probs. + consensus@8": Method("8 samples", stack(token_f, anchors=consensus_f), "stack"),
    # --- white-box and eight-sample baselines ---
    "Linear probe": Method("white-box", lambda B, H: None if H is None else H[:, 1, 1], "lr_hs"),
    "Wb-S": Method(
        "white-box",
        lambda B, H: None if H is None else np.hstack([H[:, 1, 0], H[:, 1, 2], B["BASE"], B["TRACE"][:, 16:]]),
        "lr_hs",
    ),
    "Supervised Consensus@8": Method("8 samples", lambda B, H: np.hstack([B["SC"][:, 1:3], B["TRACE"], B["FLL"][:, :1]]), "lr"),
    # --- ablations of TrAC ---
    "Abl: logistic head": Method("ablation", lambda B, H: trac_f(B), "lr"),
    "Abl: without rival answers": Method("ablation", stack(with_(token_f, block("TRAJ"), block("TFT"))), "stack"),
    "Abl: rival at the end only": Method(
        "ablation", stack(with_(token_f, block("TRAJ_4"), block("TFT"), block("CON_END")), needs=("CON_END",)), "stack"
    ),
    "Abl: without likelihood of a": Method(
        "ablation", stack(with_(token_f, block("TRAJ_4", DECODED_COLUMNS), block("CON"))), "stack"
    ),
    "Abl: end of trace only": Method("ablation", stack(with_(token_f, block("TFT"), block("TRAJ_4", [20, 41, 51]))), "stack"),
    "Abl: greedy answers only": Method("ablation", stack(with_(token_f, block("TRAJ_G"), block("TFT"), block("CON"))), "stack"),
    "Abl: 20 checkpoints": Method("ablation", stack(with_(token_f, block("TRAJ"), block("TFT"), block("CON"))), "stack"),
}


def raw_baselines(B: Blocks) -> Dict[str, Dict[str, np.ndarray]]:
    """Training-free scores (higher = more confident), grouped by budget."""
    sd, sp = B["SD"], B["SAMP"]
    V = sd[:, 1]
    grey = {
        "Output length": -np.log1p(B["META"][:, 0]),
        "Mean log-prob": B["BASE"][:, 0],
        "Self-certainty": B["BASE"][:, 1],
        "DeepConf-LGC": B["BASE"][:, 2],
        "P(True)": B["PT"][:, 0],
        "Verbalized confidence": V,
        "SelfDoubt": z(sd[:, 0]) + (z(V) if np.isfinite(V).all() else 0.0),
    }
    if "TOPK" in B:
        grey["Verbalized top-k"] = B["TOPK"][:, 5]
    if "PC" in B:
        grey["Prefix consistency"] = np.nan_to_num(B["PC"][:, 0], nan=0.5)
    judge = {}
    if "RM" in B:
        judge["Reward model"] = B["RM"][:, 0]
    if "PRM" in B:
        judge["Math PRM"] = np.nan_to_num(B["PRM"][:, 0], nan=0.5)
    samples = {
        "SC@8": B["SC"][:, 0],
        "CISC@8": sp[:, 8],
        "Semantic entropy@8": sp[:, 2],
        "Semantic entropy@8 (unweighted)": sp[:, 1],
        "SelfCheckGPT@8": sp[:, 0],
        "EigV@8": sp[:, 4],
        "Deg@8": sp[:, 6],
        "KLE@8": sp[:, 7],
    }
    return {"grey-box": grey, "judge": judge, "8 samples": samples}
