from sentiment import score_text


def test_not_good_is_negative():
    assert score_text("not good") < 0


def test_very_good_outscores_good():
    assert score_text("very good") > score_text("good")


def test_not_very_good_is_negative():
    # Regression test for the original recursive scorer, which only ever
    # looked one token back. By the time "good" was scored, the closest
    # preceding token was "very" (an intensifier), so "not" had already
    # been forgotten and the review scored positive.
    assert score_text("not very good") < 0


def test_empty_review_scores_zero():
    assert score_text("") == 0
    assert score_text("   ") == 0


def test_unknown_word_contributes_zero():
    assert score_text("xyzzyplugh") == 0
