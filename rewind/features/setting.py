"""All feature blocks of one setting (cache tag), aligned to the response ids of its feature file."""

from __future__ import annotations

from typing import Dict

import numpy as np

from rewind import config as C
from rewind.features.intermediate_answers import intermediate_answer_blocks
from rewind.features.runner_up import runner_up_blocks
from rewind.optional import optional_module


def load_setting(tag: str) -> Dict[str, np.ndarray]:
    """Blocks of rewind.features.build plus, when their caches exist, the intermediate-answer blocks
    (INTERMEDIATE*, TFT) and the runner-up blocks (RUNNER_UP, RUNNER_UP_END); with the baselines package present, also its
    per-setting blocks (baselines.features.setting_blocks). Rows are unique by question id."""
    B = dict(np.load(C.FEATURES / f"{tag}.npz"))
    ids = np.array([str(q) for q in B["ids"]])
    last = {q: i for i, q in enumerate(ids)}
    if len(last) < len(ids):  # TriviaQA repeats some question ids: keep each id's last row,
        keep = np.array(sorted(last.values()))  # the one every id-keyed query record refers to
        B = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(ids) else v) for k, v in B.items()}
    ids = B["ids"]
    B.update(intermediate_answer_blocks(tag, ids))
    B.update(runner_up_blocks(tag, ids))
    extra = optional_module("baselines.features", "the baseline feature blocks")
    if extra:
        B.update(extra.setting_blocks(tag, ids))
    return B
