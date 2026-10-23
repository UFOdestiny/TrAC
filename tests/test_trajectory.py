import numpy as np

from trac.analysis.trajectory_features import GRID, greedy_only, thin, trac0_score, trajectory_features
from trac.pipeline.probes import checkpoints, even_fractions, token_chunks


def record(agree_ck, agree_end=True, samples=None):
    T = 100
    ck = [{"k": 10 * (j + 1), "ans": "1" if a else "2", "agree": a, "tf": [-0.1, -0.2], "tok_lps": [-0.1]}
          for j, a in enumerate(agree_ck)]
    for c in ck:
        if samples is not None:
            c["samp"] = {"ans": ["1"] * samples[0] + ["2"] * samples[1], "agree": [True] * samples[0] + [False] * samples[1]}
    end = {"ans": "1", "agree": agree_end, "tf": [-0.1, -0.2], "tok_lps": [-0.1]}
    return {"T": T, "ck": ck, "end": end}


def test_trac0_bounds():
    assert trac0_score(record([True] * 9)) == 1.0
    assert trac0_score(record([False] * 9, agree_end=False)) == 0.0
    mixed = trac0_score(record([False] * 4 + [True] * 5, samples=(2, 2)))
    assert 0.0 < mixed < 1.0
    assert trac0_score(greedy_only(record([True] * 9, samples=(0, 4)))) == 1.0


def test_feature_shape_and_thinning():
    f = trajectory_features(record([True] * 9, samples=(3, 1)))
    assert f.shape == (2 * (len(GRID) + 1) + 10,)
    assert len(thin(record([True] * 9), 4)["ck"]) == 4


def test_checkpoints_snap_to_line_breaks():
    body = list(range(100))
    nl = lambda t: t % 10 == 9
    cks = checkpoints(body, even_fractions(3), nl)
    assert all(0 < k < 100 and nl(body[k - 1]) for k in cks) and cks == sorted(set(cks))


def test_token_chunks():
    items = [{"prefix": [0] * n} for n in (5, 5, 5, 20)]
    assert [len(c) for c in token_chunks(items, 12)] == [2, 1, 1]
    assert np.isclose(even_fractions(20, 4)[0], round(1 / 21, 4))
