"""Unit tests for Experiment K (exploratory) helpers; no dataset access."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln

from beat import exp_k as K
from beat.common import holm, t_greater, t_less, write_csv, write_json, write_text

PRIOR = np.array([0.5, 0.5])
GOOD = np.array([[0.9, 0.0, 0.1], [0.1, 0.0, 0.9]])
WEAK = np.array([[0.7, 0.0, 0.3], [0.3, 0.0, 0.7]])


def _stream(subject, segment, target, output):
    return K.Stream(np.asarray(subject), np.asarray(segment, np.int8),
                    np.asarray(target, np.int8), np.asarray(output, np.int8))


@pytest.mark.parametrize("variant", ["marginal", "confusion"])
def test_eq_lp_equalizes_and_is_row_stochastic(variant):
    conf = np.random.default_rng(3).dirichlet(np.ones(3), size=(5, 2))
    M, shared, optimum, cert = K.eq_lp(conf, PRIOR, variant)
    dist = K.achieved(conf, M, PRIOR, variant)
    assert np.allclose(dist, dist[0], atol=1e-9)
    assert np.allclose(M.sum(-1), 1) and M.min() >= 0
    assert cert["duality_gap"] < 1e-8 and cert["dual_infeasibility"] < 1e-8
    # Always-hold is feasible with net progress 0, so the optimum cannot be below it.
    assert optimum >= -1e-12


@pytest.mark.parametrize("variant", ["marginal", "confusion"])
def test_eq_lp_identical_users_keep_their_decoder(variant):
    M, _, optimum, _ = K.eq_lp(np.stack([GOOD] * 3), PRIOR, variant)
    assert optimum == pytest.approx(0.8)


def test_eq_lp_confusion_garbles_to_weaker_user():
    # Blackwell: the better user can be garbled to the weaker one (net 0.4) but not vice versa.
    _, _, optimum, _ = K.eq_lp(np.stack([GOOD, WEAK]), PRIOR, "confusion")
    assert optimum == pytest.approx(0.4)


def test_eq_lp_optimum_beats_random_feasible_points():
    rng = np.random.default_rng(5)
    conf = rng.dirichlet(np.ones(3), size=(3, 2))
    _, _, optimum, _ = K.eq_lp(conf, PRIOR, "marginal")
    # A common randomized-response channel equalizes only if the marginals already agree,
    # so compare with the always-hold channel and a channel that ignores the command.
    for col in range(3):
        M = np.zeros((3, 3, 3))
        M[:, :, col] = 1
        dist = K.achieved(conf, M, PRIOR, "marginal")
        assert np.allclose(dist, dist[0])
        net = K.net_progress_from(np.einsum("y,uyo->uyo", PRIOR, np.einsum("uyc,uco->uyo", conf, M)))[3]
        assert optimum >= net.mean() - 1e-12


@pytest.mark.parametrize("variant", ["marginal", "confusion"])
def test_eq_lp_weighted_utility_drops_moves_that_cost_more_than_they_gain(variant):
    w = K.wrong_weight(0.8)
    # At p = tau a move's expected weighted utility p - w (1 - p) is zero.
    assert 0.8 - w * 0.2 == pytest.approx(0.0)
    _, _, kept, _ = K.eq_lp(np.stack([GOOD] * 3), PRIOR, variant, w)
    assert kept == pytest.approx(0.9 - w * 0.1)
    M, _, held, _ = K.eq_lp(np.stack([WEAK] * 3), PRIOR, variant, w)
    assert held == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(M[:, [0, 2], 1], 1)


def test_crossfit_rows_use_matrices_fitted_without_their_fold():
    rng = np.random.default_rng(7)
    users = np.array([3, 8])
    s = _stream(np.repeat(users, 20), np.ones(40), rng.integers(2, size=40),
                rng.integers(3, size=40))
    fold, mats = K.crossfit_matrices(s, users, np.zeros(2, np.int64), "confusion", PRIOR,
                                     0.5, 4.0, 5)
    assert np.array_equal(fold, np.tile(np.repeat(np.arange(5), 4), 2))
    for f in range(5):
        expected, _ = K.fit_eq(K.confusion_counts(s.subset(fold != f), users),
                               np.zeros(2, np.int64), "confusion", PRIOR, 0.5, 4.0)
        assert np.allclose(mats[f], expected)
    # Each row is post-processed by its own fold's matrices.
    marker = np.zeros((5, 2, 3, 3))
    for f in range(5):
        marker[f, :, :, f % 3] = 1
    out = K.apply_crossfit(s, fold, marker, users, rng)
    assert np.array_equal(out.output, fold % 3)


def test_text_writers_emit_lf_only(tmp_path):
    write_json(tmp_path / "a.json", {"x": [1, 2], "y": {"z": "w"}})
    write_text(tmp_path / "b.txt", "one\ntwo\n")
    write_csv(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), tmp_path / "c.csv")
    K.write_eq_guarantee(tmp_path / "d.md")
    for name in ("a.json", "b.txt", "c.csv", "d.md"):
        data = (tmp_path / name).read_bytes()
        assert b"\n" in data and b"\r" not in data, name


def test_rr_epsilon_is_max_log_ratio():
    for q in (0.1, 0.5, 0.75):
        M = K.rr_matrix(q)
        assert np.allclose(M.sum(1), 1)
        assert K.rr_epsilon(q) == pytest.approx(np.log(M.max() / M.min()))


def test_dirmult_matches_bruteforce_up_to_user_constant():
    rng = np.random.default_rng(0)
    n = rng.integers(0, 6, size=(4, 1, 3)).astype(float)
    m = rng.integers(0, 4, size=(7, 1, 3)).astype(float)
    alpha = 0.5
    got = K.dirmult(m, n, alpha)
    a = n + alpha
    brute = np.array([[gammaln(a[u].sum()) - gammaln(a[u].sum() + w.sum())
                       + (gammaln(a[u] + w) - gammaln(a[u])).sum() for u in range(4)] for w in m])
    assert np.allclose(got - got[:, :1], brute - brute[:, :1])


def test_counts_windows_match_direct_counts():
    rng = np.random.default_rng(1)
    subject = np.repeat([1, 2], 30)
    segment = np.tile(np.repeat([1, 2], 15), 2)
    target, output = rng.integers(2, size=60), rng.integers(3, size=60)
    s = _stream(subject, segment, target, output)
    starts = K.window_starts(s, 5)
    assert all(subject[i] == subject[i + 4] and segment[i] == segment[i + 4] for i in starts)
    win = K.Counts(s).windows(starts, 5)
    for j, i in enumerate(starts):
        o, y = output[i:i + 5], target[i:i + 5]
        assert np.array_equal(win["cat"][j, 0], np.bincount(o, minlength=3))
        trans = np.zeros((3, 3))
        for a, b in zip(o[:-1], o[1:]):
            trans[a, b] += 1
        assert np.array_equal(win["trans"][j], trans)
        aware = np.zeros((2, 3))
        np.add.at(aware, (y, o), 1)
        assert np.array_equal(win["aware"][j], aware)


def test_bayes_identifies_separable_users_and_chance():
    rng = np.random.default_rng(2)
    users = np.array([0, 1, 2])
    outputs = np.repeat([0, 1, 2], 200)
    s = _stream(np.repeat(users, 200), np.tile(np.repeat([1, 2], 100), 3),
                rng.integers(2, size=600), outputs)
    enrol = K.Counts(s.subset(s.segment == 1)).enrol(users)
    test = s.subset(s.segment == 2)
    starts = K.window_starts(test, 5)
    win = K.Counts(test).windows(starts, 5)
    scores = K.bayes_scores("cat_blind", win, enrol, 0.5)
    groups = {"g": np.array([0, 0, 1])}
    rec = K.recalls(scores, users, test.subject[starts], groups, rng)
    assert np.allclose(rec["nway"], 1) and np.allclose(rec["g"], 1)
    ch = K.chance(users, groups)
    assert np.allclose(ch["nway"], 1 / 3) and np.allclose(ch["g"], [0.5, 0.5, 1.0])


def test_identity_channel_keeps_outputs():
    s = _stream([0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 2])
    out = K.apply_channel(s, np.repeat(np.eye(3)[None], 2, axis=0), np.array([0, 1]),
                          np.random.default_rng(0))
    assert np.array_equal(out.output, s.output)


def test_make_groups_sizes_and_order():
    users = np.arange(10)
    utility = np.arange(10)[::-1].astype(float)
    group = K.make_groups(users, utility, 3)
    assert sorted(np.bincount(group)) == [3, 3, 4]
    assert group[9] == 0 and group[0] == group[1] == group[2] == 2


def test_t_tests_match_scipy():
    x = np.random.default_rng(4).normal(0.1, 1, size=30)
    assert t_greater(x, 0.0) == pytest.approx(stats.ttest_1samp(x, 0, alternative="greater").pvalue)
    assert t_less(x, 0.5) == pytest.approx(stats.ttest_1samp(x, 0.5, alternative="less").pvalue)


def test_holm_step_down():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_m3_and_m4_pvalues_respond_to_signal():
    rng = np.random.default_rng(6)
    n = 40
    strong = rng.normal(0.5, 0.1, size=(4, n))
    base = rng.normal(0.05, 0.02, size=(4, n))
    assert K.m3_pvalue(strong, base, n, 2.0) < 1e-6
    assert K.m3_pvalue(base, base, n, 2.0) > 0.05
    D = rng.normal(0.0, 0.02, size=(4, n))
    D0 = rng.normal(0.4, 0.05, size=(4, n))
    loss = rng.normal(0.0, 0.01, size=n)
    p, parts = K.m4_pvalue(D, D0, loss, 0.05, 0.05, 0.02)
    assert p < 1e-3 and set(parts) == {"bound", "reduction", "utility"}
    assert K.m4_pvalue(D0, D0, loss, 0.05, 0.05, 0.02)[0] > 0.05
