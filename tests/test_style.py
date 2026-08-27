"""Tests for style profiling and comparison.

All sample data is fictional and hand-written.
"""

from mini_gabriel.style import (
    compare,
    render_comparison,
    self_distance,
    style_profile,
)

# Two deliberately different writing styles, used throughout.
TERSE = [["ya"], ["ok lah"], ["cannot sia"], ["ya", "6pm"], ["k"], ["wah damn"]]
FORMAL = [
    ["Yes, that works for me."],
    ["I will be there at six."],
    ["Unfortunately I cannot make it."],
    ["That sounds like a good plan."],
    ["Certainly, see you then."],
    ["I appreciate the invitation."],
]


class TestProfileBasics:
    def test_empty_input(self):
        assert style_profile([])["n"] == 0

    def test_counts_replies_not_messages(self):
        assert style_profile([["a", "b", "c"], ["d"]])["n"] == 2

    def test_blank_replies_are_ignored(self):
        assert style_profile([[], ["hi"]])["n"] == 1

    def test_mean_messages_reflects_bursts(self):
        assert style_profile([["a", "b"], ["c", "d"]])["mean_messages"] == 2.0

    def test_multi_message_rate(self):
        assert style_profile([["a", "b"], ["c"], ["d"], ["e"]])["multi_message_rate"] == 0.25


class TestStyleMarkers:
    def test_lowercase_start_is_detected(self):
        assert style_profile([["ya"], ["ok"]])["lowercase_start_rate"] == 1.0
        assert style_profile([["Yes"], ["Okay"]])["lowercase_start_rate"] == 0.0

    def test_terminal_punctuation_is_detected(self):
        assert style_profile([["done."], ["really?"]])["terminal_punct_rate"] == 1.0
        assert style_profile([["done"], ["really"]])["terminal_punct_rate"] == 0.0

    def test_trivial_replies_are_detected(self):
        assert style_profile([["ok"], ["lol"]])["trivial_rate"] == 1.0

    def test_a_burst_with_one_real_message_is_not_trivial(self):
        assert style_profile([["ok", "ill be there at six"]])["trivial_rate"] == 0.0

    def test_emoji_is_detected(self):
        assert style_profile([["nice 🔥"]])["emoji_rate"] == 1.0
        assert style_profile([["nice"]])["emoji_rate"] == 0.0

    def test_singlish_is_detected(self):
        assert style_profile([["cannot lah"], ["walao"]])["singlish_rate"] == 1.0
        assert style_profile([["I cannot attend"]])["singlish_rate"] == 0.0

    def test_questions_are_detected(self):
        assert style_profile([["what time?"], ["ok"]])["question_rate"] == 0.5

    def test_length_uses_the_whole_burst(self):
        # Messages are joined with a newline, so a burst is measured as one reply.
        assert style_profile([["abc", "de"]])["mean_chars"] == 6.0


class TestComparison:
    def test_identical_profiles_score_zero(self):
        profile = style_profile(TERSE)
        assert compare(profile, profile)["style_distance"] == 0.0

    def test_different_styles_score_high(self):
        result = compare(style_profile(TERSE), style_profile(FORMAL))
        assert result["style_distance"] > 0.3

    def test_comparison_reports_direction(self):
        result = compare(style_profile(TERSE), style_profile(FORMAL))
        # The formal set is longer and capitalised, so both differences are signed.
        assert result["metrics"]["mean_chars"]["difference"] > 0
        assert result["metrics"]["lowercase_start_rate"]["difference"] < 0

    def test_a_formal_model_is_caught_by_the_obvious_markers(self):
        result = compare(style_profile(TERSE), style_profile(FORMAL))
        assert result["metrics"]["lowercase_start_rate"]["normalised"] > 0.9
        assert result["metrics"]["terminal_punct_rate"]["normalised"] > 0.9

    def test_metrics_missing_from_one_side_are_skipped(self):
        result = compare(style_profile(TERSE), {"n": 0})
        assert result["metrics"] == {}
        assert result["style_distance"] is None

    def test_relative_difference_is_capped(self):
        # A wildly long candidate must not swamp the average.
        result = compare(style_profile([["hi"]]), style_profile([["x" * 100_000]]))
        assert result["metrics"]["mean_chars"]["normalised"] == 1.0

    def test_sample_sizes_are_reported(self):
        result = compare(style_profile(TERSE), style_profile(FORMAL[:2]))
        assert result["reference_n"] == 6
        assert result["candidate_n"] == 2


class TestSelfDistance:
    def test_same_style_scores_near_zero(self):
        # The floor: consistent writing compared against itself.
        assert self_distance(TERSE * 20) < 0.05

    def test_floor_is_below_a_cross_style_comparison(self):
        floor = self_distance(TERSE * 20)
        cross = compare(style_profile(TERSE), style_profile(FORMAL))["style_distance"]
        assert floor < cross

    def test_tiny_samples_return_zero_rather_than_noise(self):
        assert self_distance([["hi"]]) == 0.0

    def test_ordering_does_not_inflate_the_floor(self):
        # Perfectly alternating input is the adversarial case: taking every
        # other item would separate the two styles completely and report a huge
        # floor. A shuffled split keeps both halves representative.
        mixed = []
        for terse, formal in zip(TERSE * 10, FORMAL * 10):
            mixed.extend([terse, formal])
        assert self_distance(mixed) < 0.2

    def test_floor_is_reproducible(self):
        assert self_distance(TERSE * 20) == self_distance(TERSE * 20)


class TestRendering:
    def test_render_contains_metrics_and_distance(self):
        text = render_comparison(compare(style_profile(TERSE), style_profile(FORMAL)))
        assert "style distance" in text
        assert "lowercase_start_rate" in text

    def test_render_contains_no_message_text(self):
        canary = "FICTIONAL-CANARY-77x"
        result = compare(style_profile([[canary]]), style_profile([[canary]]))
        assert canary not in render_comparison(result)
