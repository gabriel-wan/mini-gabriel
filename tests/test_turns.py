"""Tests for turn and session segmentation.

All sample data is fictional and hand-written.
"""

from mini_gabriel.turns import (
    DEFAULT_BURST_GAP_SECONDS,
    Turn,
    build_turns,
    has_usable_text,
    split_sessions,
)

BASE = "2026-03-01T12:00:00+00:00"


def msg(message_id, text, mine=True, sender=1, minute=0, second=0, **extra):
    record = {
        "message_id": message_id,
        "text": text,
        "is_outgoing": mine,
        "sender_id": sender,
        "date_utc": f"2026-03-01T12:{minute:02d}:{second:02d}+00:00",
    }
    record.update(extra)
    return record


def turn(*messages, mine=True, sender=1, start=BASE, end=BASE):
    return Turn(sender_id=sender, is_me=mine, messages=tuple(messages), start_utc=start, end_utc=end)


class TestUsableText:
    def test_text_counts(self):
        assert has_usable_text({"text": "hi"})

    def test_empty_and_whitespace_do_not(self):
        assert not has_usable_text({"text": ""})
        assert not has_usable_text({"text": "   "})
        assert not has_usable_text({})


class TestBuildTurns:
    def test_rapid_messages_merge_into_one_turn(self):
        turns = build_turns([msg(1, "ya", second=0), msg(2, "6pm", second=8)])
        assert len(turns) == 1
        assert turns[0].messages == ("ya", "6pm")

    def test_speaker_change_ends_a_turn(self):
        turns = build_turns([
            msg(1, "you coming", mine=False, sender=2),
            msg(2, "ya", mine=True, sender=1),
        ])
        assert len(turns) == 2
        assert turns[0].is_me is False
        assert turns[1].is_me is True

    def test_long_pause_ends_a_turn_even_for_same_speaker(self):
        turns = build_turns([msg(1, "ya", minute=0), msg(2, "actually no", minute=30)])
        assert len(turns) == 2

    def test_gap_exactly_at_the_limit_still_merges(self):
        turns = build_turns([
            msg(1, "a", second=0),
            msg(2, "b", minute=DEFAULT_BURST_GAP_SECONDS // 60, second=0),
        ])
        assert len(turns) == 1

    def test_two_people_in_a_group_never_merge(self):
        turns = build_turns([
            msg(1, "hi", mine=False, sender=2, second=0),
            msg(2, "yo", mine=False, sender=3, second=1),
        ])
        assert len(turns) == 2

    def test_media_is_skipped_without_splitting_the_burst(self):
        # A photo mid-burst is part of the same stretch of typing.
        turns = build_turns([
            msg(1, "haha", second=0),
            msg(2, "", second=2, has_media=True, media_type="MessageMediaPhoto"),
            msg(3, "look at this", second=4),
        ])
        assert len(turns) == 1
        assert turns[0].messages == ("haha", "look at this")

    def test_records_are_sorted_by_id(self):
        turns = build_turns([msg(2, "second", second=8), msg(1, "first", second=0)])
        assert turns[0].messages == ("first", "second")

    def test_whitespace_is_stripped(self):
        turns = build_turns([msg(1, "  padded  ")])
        assert turns[0].messages == ("padded",)

    def test_empty_input(self):
        assert build_turns([]) == []

    def test_only_media_produces_no_turns(self):
        assert build_turns([msg(1, "", has_media=True)]) == []

    def test_timestamps_span_the_burst(self):
        turns = build_turns([msg(1, "a", second=0), msg(2, "b", second=30)])
        assert turns[0].start_utc.endswith("12:00:00+00:00")
        assert turns[0].end_utc.endswith("12:00:30+00:00")


class TestTurnProperties:
    def test_text_joins_with_newlines(self):
        assert turn("ya", "6pm").text == "ya\n6pm"

    def test_all_short_messages_are_trivial(self):
        assert turn("ok").is_trivial
        assert turn("ok", "lol").is_trivial

    def test_one_substantive_message_makes_the_turn_not_trivial(self):
        # This is why the per-turn rate is half the per-message rate.
        assert not turn("ok", "ill be there at 6").is_trivial

    def test_six_characters_is_not_trivial(self):
        assert not turn("okayyy").is_trivial


class TestSessions:
    def test_close_turns_stay_in_one_session(self):
        turns = build_turns([
            msg(1, "hi", mine=False, sender=2, minute=0),
            msg(2, "yo", minute=1),
        ])
        assert len(split_sessions(turns)) == 1

    def test_long_silence_starts_a_new_session(self):
        turns = [
            turn("morning", start="2026-03-01T02:00:00+00:00", end="2026-03-01T02:00:00+00:00"),
            turn("evening", start="2026-03-01T22:00:00+00:00", end="2026-03-01T22:00:00+00:00"),
        ]
        assert len(split_sessions(turns)) == 2

    def test_gap_under_the_threshold_does_not_split(self):
        turns = [
            turn("a", start="2026-03-01T02:00:00+00:00", end="2026-03-01T02:00:00+00:00"),
            turn("b", start="2026-03-01T04:00:00+00:00", end="2026-03-01T04:00:00+00:00"),
        ]
        assert len(split_sessions(turns)) == 1

    def test_silence_is_measured_from_the_end_of_the_previous_turn(self):
        turns = [
            turn("long burst", start="2026-03-01T02:00:00+00:00", end="2026-03-01T05:00:00+00:00"),
            turn("reply", start="2026-03-01T06:00:00+00:00", end="2026-03-01T06:00:00+00:00"),
        ]
        # 1 hour after the burst ended, not 4 hours after it started.
        assert len(split_sessions(turns)) == 1

    def test_empty_input(self):
        assert split_sessions([]) == []

    def test_custom_threshold(self):
        turns = [
            turn("a", start="2026-03-01T02:00:00+00:00", end="2026-03-01T02:00:00+00:00"),
            turn("b", start="2026-03-01T03:00:00+00:00", end="2026-03-01T03:00:00+00:00"),
        ]
        assert len(split_sessions(turns, session_gap_seconds=600)) == 2
