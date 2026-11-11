"""Feature views of REWIND, from the blocks of rewind.features.setting.load_setting:

  token-probability features phi_P
    TRACE   binned token log-probabilities and entropies of the original generation, with trends
    ANS     confidence over the answer section and the boxed answer
    META    output length, reasoning closed, truncated
  query features phi_A
    INTERMEDIATE_4  intermediate answers at four checkpoints and the end of the trace, with the level and
            late-half rise of the likelihood of a
    TFT     likelihood of the final answer a at the end of the trace
    RUNNER_UP     runner-up answers: share and margin of a against the model's own alternatives
  sample-consensus features phi_K (REWIND@8, meta-classifier inputs)
    SC, SAMP  vote shares and answer entropy of the K traces, plain and likelihood-weighted
"""

from __future__ import annotations

import numpy as np

from rewind.classifier import logit


def length_f(B):
    """Log output length, reasoning closed, truncated."""
    return np.hstack([np.log1p(B["META"][:, :1]), B["META"][:, 1:]])


def token_f(B):
    """Token-probability features phi_P."""
    return np.hstack([B["TRACE"], B["ANS"], length_f(B)])


def answer_f(B):
    """Query features phi_A: intermediate answers, likelihood of a, and runner-up answers."""
    return np.hstack([B["INTERMEDIATE_4"], B["TFT"], B["RUNNER_UP"]])


def rewind_f(B):
    """REWIND: phi = [phi_P; phi_A]."""
    return np.hstack([token_f(B), answer_f(B)])


def lite_f(B):
    """REWIND-Lite: greedy answers at four checkpoints, likelihood of a, runner-up at the end."""
    return np.hstack([token_f(B), B["INTERMEDIATE_G4"], B["TFT"], B["RUNNER_UP_END"]])


def consensus_f(B):
    """Sample-consensus features phi_K of REWIND@8: logit vote share of a and of the modal answer,
    answer entropy, logit likelihood-weighted vote share of a, and likelihood-weighted entropy."""
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
