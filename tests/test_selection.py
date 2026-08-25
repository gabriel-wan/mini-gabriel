"""Tests for the pure selection logic.

Every sample value in this file is fictional and hand-written. Nothing here
comes from a real Telegram account, and no test touches the network.
"""

from datetime import datetime, timezone

import pytest

from mini_gabriel.selection import (
    BROADCAST,
    GROUP,
    PRIVATE,
    SUPERGROUP,
    DialogDescriptor,
    SelectionCriteria,
    aggregate_chat_stats,
    descriptor_from_manifest_entry,
    evaluate_chat,
    evaluate_dialog,
    has_text,
    is_in_window,
    year_window_utc,
)


def make_message(is_outgoing=True, text="a fictional message", date_utc="2026-03-01T04:00:00+00:00", **extra):
    record = {"is_outgoing": is_outgoing, "text": text, "date_utc": date_utc}
    record.update(extra)
    return record


def make_dialog(**overrides):
    defaults = dict(chat_id=1001, name="Fictional Chat", chat_type=PRIVATE)
    defaults.update(overrides)
    return DialogDescriptor(**defaults)


class TestYearWindow:
    def test_singapore_year_begins_on_previous_utc_day(self):
        start, end = year_window_utc(2026, "Asia/Singapore")
        assert start == datetime(2025, 12, 31, 16, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 12, 31, 16, 0, tzinfo=timezone.utc)

    def test_singapore_window_starts_before_utc_window(self):
        singapore_start, _ = year_window_utc(2026, "Asia/Singapore")
        utc_start, _ = year_window_utc(2026, "UTC")
        assert singapore_start < utc_start

    def test_window_is_half_open(self):
        start, end = year_window_utc(2026, "Asia/Singapore")
        assert is_in_window(start, start, end)
        assert not is_in_window(end, start, end)

    def test_new_year_message_counts_as_target_year(self):
        # 2026-01-01 00:30 in Singapore is 2025-12-31 16:30 UTC. Treating the
        # window as UTC would wrongly exclude it.
        start, end = year_window_utc(2026, "Asia/Singapore")
        just_after_midnight_sgt = datetime(2025, 12, 31, 16, 30, tzinfo=timezone.utc)
        assert is_in_window(just_after_midnight_sgt, start, end)

    def test_late_december_message_still_counts(self):
        start, end = year_window_utc(2026, "Asia/Singapore")
        # 2026-12-31 23:00 SGT == 2026-12-31 15:00 UTC, still inside the year.
        assert is_in_window(datetime(2026, 12, 31, 15, 0, tzinfo=timezone.utc), start, end)
        # 2027-01-01 00:30 SGT == 2026-12-31 16:30 UTC, outside it.
        assert not is_in_window(datetime(2026, 12, 31, 16, 30, tzinfo=timezone.utc), start, end)

    def test_naive_datetime_is_rejected(self):
        start, end = year_window_utc(2026, "Asia/Singapore")
        with pytest.raises(ValueError):
            is_in_window(datetime(2026, 6, 1, 12, 0), start, end)


class TestHasText:
    @pytest.mark.parametrize("text", ["hello", "  padded  ", "ok"])
    def test_real_text_counts(self, text):
        assert has_text({"text": text})

    @pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
    def test_empty_text_does_not_count(self, text):
        assert not has_text({"text": text})

    def test_missing_key_does_not_count(self):
        assert not has_text({})


class TestDialogEligibility:
    def test_private_chat_is_extracted(self):
        assert evaluate_dialog(make_dialog(chat_type=PRIVATE)).included

    def test_group_is_extracted(self):
        assert evaluate_dialog(make_dialog(chat_type=GROUP)).included

    def test_supergroup_is_extracted(self):
        # Supergroups are Channel objects in Telethon but must not be treated
        # as broadcast channels.
        assert evaluate_dialog(make_dialog(chat_type=SUPERGROUP)).included

    def test_broadcast_channel_is_excluded(self):
        decision = evaluate_dialog(make_dialog(chat_type=BROADCAST))
        assert not decision.included
        assert "broadcast channel" in decision.reason_text

    def test_bot_is_excluded(self):
        decision = evaluate_dialog(make_dialog(is_bot=True))
        assert not decision.included
        assert "bot" in decision.reason_text

    def test_saved_messages_is_excluded(self):
        decision = evaluate_dialog(make_dialog(is_self_chat=True))
        assert not decision.included
        assert "saved messages" in decision.reason_text

    def test_unknown_chat_type_is_excluded(self):
        assert not evaluate_dialog(make_dialog(chat_type="mystery")).included

    def test_size_is_not_considered_at_this_stage(self):
        # Extraction must not pre-filter on size; that is the analysis stage's
        # job and cannot be known before fetching.
        huge = make_dialog(chat_type=SUPERGROUP, participant_count=5000)
        assert evaluate_dialog(huge).included

    def test_multiple_reasons_are_collected(self):
        decision = evaluate_dialog(make_dialog(chat_type=BROADCAST, is_bot=True))
        assert len(decision.reasons) == 2


class TestAggregation:
    def test_counts_split_by_author(self):
        messages = [
            make_message(is_outgoing=True),
            make_message(is_outgoing=True),
            make_message(is_outgoing=False),
        ]
        stats = aggregate_chat_stats(make_dialog(), messages)
        assert stats.total_messages == 3
        assert stats.my_messages == 2
        assert stats.other_messages == 1

    def test_media_only_message_counts_as_message_but_not_as_text(self):
        messages = [
            make_message(text="an actual sentence"),
            make_message(text="", has_media=True, media_type="MessageMediaPhoto"),
            make_message(text="   ", is_service=True),
        ]
        stats = aggregate_chat_stats(make_dialog(), messages)
        assert stats.my_messages == 3
        assert stats.my_text_messages == 1

    def test_date_range_uses_earliest_and_latest(self):
        messages = [
            make_message(date_utc="2026-06-01T00:00:00+00:00"),
            make_message(date_utc="2026-01-05T00:00:00+00:00"),
            make_message(date_utc="2026-11-30T00:00:00+00:00"),
        ]
        stats = aggregate_chat_stats(make_dialog(), messages)
        assert stats.first_message_utc == "2026-01-05T00:00:00+00:00"
        assert stats.last_message_utc == "2026-11-30T00:00:00+00:00"

    def test_empty_chat_aggregates_to_zero(self):
        stats = aggregate_chat_stats(make_dialog(), [])
        assert stats.total_messages == 0
        assert stats.first_message_utc is None

    def test_accepts_a_generator(self):
        stats = aggregate_chat_stats(make_dialog(), (make_message() for _ in range(5)))
        assert stats.total_messages == 5


class TestChatQualification:
    def build_stats(self, **overrides):
        dialog = make_dialog(
            chat_type=overrides.pop("chat_type", PRIVATE),
            participant_count=overrides.pop("participant_count", 2),
            participant_count_known=overrides.pop("participant_count_known", True),
        )
        my_text = overrides.pop("my_text_messages", 150)
        messages = [make_message(text="fictional") for _ in range(my_text)]
        return aggregate_chat_stats(dialog, messages, **overrides)

    def test_busy_private_chat_qualifies(self):
        assert evaluate_chat(self.build_stats(my_text_messages=150)).included

    def test_exactly_at_threshold_qualifies(self):
        assert evaluate_chat(self.build_stats(my_text_messages=100)).included

    def test_one_below_threshold_does_not(self):
        decision = evaluate_chat(self.build_stats(my_text_messages=99))
        assert not decision.included
        assert "99" in decision.reason_text

    def test_small_group_qualifies(self):
        stats = self.build_stats(chat_type=GROUP, participant_count=20)
        assert evaluate_chat(stats).included

    def test_group_over_limit_is_rejected(self):
        stats = self.build_stats(chat_type=SUPERGROUP, participant_count=21)
        decision = evaluate_chat(stats)
        assert not decision.included
        assert "exceeds limit" in decision.reason_text

    def test_unknown_participant_count_never_silently_qualifies(self):
        stats = self.build_stats(
            chat_type=SUPERGROUP, participant_count=None, participant_count_known=False
        )
        decision = evaluate_chat(stats)
        assert not decision.included
        assert "could not be determined" in decision.reason_text

    def test_private_chat_ignores_participant_count(self):
        stats = self.build_stats(
            chat_type=PRIVATE, participant_count=None, participant_count_known=False
        )
        assert evaluate_chat(stats).included

    def test_incomplete_extraction_is_flagged(self):
        stats = self.build_stats(extraction_complete=False)
        decision = evaluate_chat(stats)
        assert not decision.included
        assert "incomplete" in decision.reason_text

    def test_custom_criteria_are_honoured(self):
        stats = self.build_stats(chat_type=GROUP, participant_count=50, my_text_messages=10)
        loose = SelectionCriteria(max_participants=100, min_my_text_messages=5)
        assert evaluate_chat(stats, loose).included

    def test_all_failures_are_reported_together(self):
        stats = self.build_stats(
            chat_type=SUPERGROUP, participant_count=500, my_text_messages=3
        )
        assert len(evaluate_chat(stats).reasons) == 2


class TestDescriptorRoundTrip:
    def test_rebuilds_from_manifest_entry(self):
        entry = {
            "chat_id": -42,
            "name": "Fictional Group",
            "chat_type": SUPERGROUP,
            "participant_count": 8,
            "participant_count_known": True,
            "is_bot": False,
            "is_self_chat": False,
        }
        descriptor = descriptor_from_manifest_entry(entry)
        assert descriptor.chat_id == -42
        assert descriptor.chat_type == SUPERGROUP
        assert descriptor.participant_count == 8

    def test_defaults_when_fields_missing(self):
        descriptor = descriptor_from_manifest_entry({"chat_id": 7})
        assert descriptor.chat_type == PRIVATE
        assert descriptor.participant_count is None


class TestMigratedGroups:
    """A legacy group upgraded to a supergroup leaves a deactivated stub."""

    def test_migrated_stub_is_not_extracted(self):
        decision = evaluate_dialog(make_dialog(chat_type=GROUP, is_migrated=True))
        assert not decision.included
        assert "migrated" in decision.reason_text

    def test_live_group_is_unaffected(self):
        assert evaluate_dialog(make_dialog(chat_type=GROUP, is_migrated=False)).included

    def test_manifest_round_trip_keeps_the_flag(self):
        descriptor = descriptor_from_manifest_entry(
            {"chat_id": -5, "chat_type": GROUP, "is_migrated": True}
        )
        assert descriptor.is_migrated is True


class TestZeroParticipantCount:
    """Zero is not a real member count; it must not pass the size limit."""

    def build_group_stats(self, participant_count, known=True):
        dialog = make_dialog(
            chat_type=SUPERGROUP,
            participant_count=participant_count,
            participant_count_known=known,
        )
        return aggregate_chat_stats(dialog, [make_message(text="fictional") for _ in range(200)])

    def test_zero_members_does_not_qualify(self):
        decision = evaluate_chat(self.build_group_stats(0))
        assert not decision.included
        assert "could not be determined" in decision.reason_text

    def test_negative_members_does_not_qualify(self):
        assert not evaluate_chat(self.build_group_stats(-1)).included

    def test_one_member_group_is_rejected_as_self_only_not_as_unknown(self):
        # 0 and 1 are both rejected but for different reasons: 0 means the count
        # is missing, 1 means the group genuinely contains only me.
        decision = evaluate_chat(self.build_group_stats(1))
        assert not decision.included
        assert "no other participants" in decision.reason_text
        assert "could not be determined" not in decision.reason_text

    def test_normal_small_group_unaffected(self):
        assert evaluate_chat(self.build_group_stats(6)).included


class TestSelfOnlyGroups:
    """A group containing only me is a notepad, not a conversation."""

    def test_one_member_group_is_not_extracted(self):
        decision = evaluate_dialog(
            make_dialog(chat_type=GROUP, participant_count=1, participant_count_known=True)
        )
        assert not decision.included
        assert "no other participants" in decision.reason_text

    def test_two_member_group_is_extracted(self):
        assert evaluate_dialog(
            make_dialog(chat_type=GROUP, participant_count=2, participant_count_known=True)
        ).included

    def test_private_chat_is_never_affected(self):
        assert evaluate_dialog(
            make_dialog(chat_type=PRIVATE, participant_count=2, participant_count_known=True)
        ).included

    def test_unknown_count_defers_the_decision(self):
        # The first pass runs before the count is resolved; it must not guess.
        assert evaluate_dialog(
            make_dialog(chat_type=GROUP, participant_count=None, participant_count_known=False)
        ).included

    def test_one_member_group_does_not_qualify_at_analysis_either(self):
        dialog = make_dialog(chat_type=GROUP, participant_count=1, participant_count_known=True)
        stats = aggregate_chat_stats(dialog, [make_message(text="note to self") for _ in range(500)])
        decision = evaluate_chat(stats)
        assert not decision.included
        assert "no other participants" in decision.reason_text
