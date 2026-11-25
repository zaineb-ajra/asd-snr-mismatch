import numpy as np
import pytest

from noisemismatch.data.identity import (
    LOW_MARGIN,
    LOW_SCORE,
    LOW_SUPPORT,
    NO_P6_MATCH,
    NOT_MUTUAL,
    OK,
    MatchRule,
    accept_matches,
    match_identities,
)

RULE = MatchRule(min_score=0.10, min_margin=0.05, min_support=0.05)


def test_clear_permutation_is_accepted():
    score = np.array([[0.02, 0.60, 0.01], [0.50, 0.03, 0.02], [0.01, 0.02, 0.40]])
    result = accept_matches(score, RULE)
    assert result["col"].tolist() == [1, 0, 2]
    assert result["accepted"].all()
    assert result["reason"].eq(OK).all()
    assert result["margin"].tolist() == pytest.approx([0.57, 0.47, 0.38])


def test_each_criterion_can_reject():
    # Row 0: best is 0.30 but column 0 prefers row 1 -> not a mutual best match.
    score = np.array([[0.30, 0.00, 0.00], [0.40, 0.20, 0.00], [0.00, 0.00, 0.08]])
    result = accept_matches(score, RULE)
    reasons = dict(zip(range(3), result["reason"], strict=True))
    assert reasons[2] == LOW_SCORE  # 0.08 < 0.10
    assert NOT_MUTUAL in reasons.values()

    close = np.array([[0.30, 0.27], [0.00, 0.50]])
    assert accept_matches(close, RULE)["reason"].tolist()[0] == LOW_MARGIN

    support_low = np.array([[0.50, 0.00], [0.00, 0.50]])
    support = (np.array([[0.02, 0.0], [0.0, 0.5]]),)
    reasons = accept_matches(support_low, RULE, support=support)["reason"].tolist()
    assert reasons == [LOW_SUPPORT, OK]


def test_margin_uses_the_column_competitor():
    # Row 0 clearly prefers column 0, but row 1 scores 0.58 on column 0 too.
    score = np.array([[0.60, 0.00], [0.58, 0.30]])
    result = accept_matches(score, RULE)
    assert result.loc[0, "reason"] == LOW_MARGIN
    assert result.loc[0, "margin"] == pytest.approx(0.02)


def test_match_identities_chains_p6_then_m6():
    rng = np.random.default_rng(0)
    n = 5
    noise = lambda: rng.uniform(-0.02, 0.02, size=(n, n))  # noqa: E731
    s_p6 = noise() + 0.6 * np.eye(n)  # names faithful at +6 dB
    perm = np.array([1, 0, 3, 2, 4])  # reference i is file perm[i] at -6 dB
    s_0m = noise() + 0.3 * np.eye(n)[perm]
    s_6m = noise() + 0.4 * np.eye(n)[perm]
    s_p6[4] = noise()[4]  # reference 4 has no +6 dB counterpart
    result = match_identities(s_p6, s_0m, s_6m, RULE)
    assert result["p6_accepted"].tolist() == [True, True, True, True, False]
    assert result.loc[:3, "m6_col"].tolist() == perm[:4].tolist()
    assert result.loc[:3, "m6_argmax_agree"].all()
    assert result.loc[4, "m6_reason"] == NO_P6_MATCH
    assert result["kept"].tolist() == [True, True, True, True, False]
