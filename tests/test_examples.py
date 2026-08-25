"""Tests for training-example construction.

All sample data is fictional and hand-written.
"""

from mini_gabriel.examples import (
    ME,
    BuildConfig,
    assign_pseudonyms,
    build_examples,
    keep_trivial_turn,
    select_holdout_chats,
    take_context,
)
from mini_gabriel.turns import Turn


def turn(*messages, mine=True, sender=1, hour=12, minute=0):
    stamp = f"2026-03-01T{hour:02d}:{minute:02d}:00+00:00"
    return Turn(
        sender_id=sender, is_me=mine, messages=tuple(messages), start_utc=stamp, end_utc=stamp
    )


def conversation(pairs):
    """[(mine, sender, text), ...] -> list of turns, one minute apart."""
    return [
        turn(text, mine=mine, sender=sender, minute=i)
        for i, (mine, sender, text) in enumerate(pairs)
    ]


class TestPseudonyms:
    def test_i_am_always_me(self):
        mapping = assign_pseudonyms([turn("hi", mine=True, sender=1)])
        assert mapping[1] == ME

    def test_others_are_lettered_in_order_of_appearance(self):
        turns = conversation([(False, 2, "hi"), (False, 3, "yo"), (True, 1, "hey")])
        mapping = assign_pseudonyms(turns)
        assert mapping[2] == "A"
        assert mapping[3] == "B"
        assert mapping[1] == ME

    def test_same_person_always_gets_the_same_label(self):
        turns = conversation([(False, 2, "a"), (False, 3, "b"), (False, 2, "c")])
        mapping = assign_pseudonyms(turns)
        assert mapping[2] == "A"

    def test_no_real_identifiers_survive_into_the_labels(self):
        mapping = assign_pseudonyms(conversation([(False, 987654321, "hi")]))
        assert "987654321" not in "".join(mapping.values())

    def test_more_than_twenty_six_participants(self):
        turns = [turn("hi", mine=False, sender=100 + i, minute=i) for i in range(28)]
        labels = list(assign_pseudonyms(turns).values())
        assert labels[25] == "Z"
        assert labels[26] == "AA"


class TestTrivialDownsampling:
    def test_same_input_always_gives_the_same_answer(self):
        args = (123, "2026-03-01T12:00:00+00:00:4", 0.33, "seed")
        assert keep_trivial_turn(*args) == keep_trivial_turn(*args)

    def test_rate_of_one_keeps_everything(self):
        assert all(keep_trivial_turn(1, f"k{i}", 1.0, "s") for i in range(50))

    def test_rate_of_zero_keeps_nothing(self):
        assert not any(keep_trivial_turn(1, f"k{i}", 0.0, "s") for i in range(50))

    def test_rate_is_approximately_honoured(self):
        kept = sum(1 for i in range(4000) if keep_trivial_turn(7, f"k{i}", 0.33, "s"))
        assert 0.30 < kept / 4000 < 0.36

    def test_different_seeds_select_differently(self):
        a = [keep_trivial_turn(1, f"k{i}", 0.5, "seed-a") for i in range(200)]
        b = [keep_trivial_turn(1, f"k{i}", 0.5, "seed-b") for i in range(200)]
        assert a != b


class TestContext:
    def test_context_is_chronological(self):
        history = conversation([(False, 2, "one"), (True, 1, "two"), (False, 2, "three")])
        got = take_context(history, assign_pseudonyms(history), BuildConfig())
        assert [c["text"] for c in got] == ["one", "two", "three"]

    def test_turn_cap_keeps_the_most_recent(self):
        history = conversation([(False, 2, f"m{i}") for i in range(20)])
        got = take_context(history, assign_pseudonyms(history), BuildConfig(context_turns=3))
        assert [c["text"] for c in got] == ["m17", "m18", "m19"]

    def test_character_budget_truncates_further(self):
        history = conversation([(False, 2, "x" * 100) for _ in range(10)])
        config = BuildConfig(context_turns=10, context_token_budget=50)  # 200 chars
        got = take_context(history, assign_pseudonyms(history), config)
        assert 0 < len(got) < 10

    def test_one_oversized_turn_is_still_kept(self):
        # Otherwise an example with a single long preceding message would have
        # no context at all and be silently dropped.
        history = conversation([(False, 2, "x" * 10_000)])
        config = BuildConfig(context_token_budget=10)
        assert len(take_context(history, assign_pseudonyms(history), config)) == 1

    def test_speakers_are_pseudonymised(self):
        history = conversation([(False, 2, "hi"), (True, 1, "yo")])
        got = take_context(history, assign_pseudonyms(history), BuildConfig())
        assert {c["speaker"] for c in got} == {"A", ME}

    def test_empty_history(self):
        assert take_context([], {}, BuildConfig()) == []


class TestBuildExamples:
    def build(self, turns, **kwargs):
        # Keep every trivial turn unless a test is specifically about
        # downsampling: short messages like "yo" are trivial by definition, so
        # the default rate would randomly drop the structure under test.
        kwargs.setdefault("trivial_keep_rate", 1.0)
        config = BuildConfig(**kwargs)
        return build_examples(1, "private", [turns], assign_pseudonyms(turns), config)

    def test_only_my_turns_become_targets(self):
        turns = conversation([(False, 2, "hi"), (True, 1, "yo"), (False, 2, "sup")])
        examples, _ = self.build(turns)
        assert len(examples) == 1
        assert examples[0]["target"] == ["yo"]

    def test_target_keeps_message_boundaries(self):
        turns = [turn("hi", mine=False, sender=2, minute=0), turn("ya", "6pm", minute=1)]
        examples, _ = self.build(turns)
        assert examples[0]["target"] == ["ya", "6pm"]

    def test_turn_with_no_preceding_context_is_dropped(self):
        turns = conversation([(True, 1, "first thing i said")])
        examples, counters = self.build(turns)
        assert examples == []
        assert counters["no_context"] == 1

    def test_every_later_turn_of_mine_becomes_an_example(self):
        turns = conversation([
            (False, 2, "a"), (True, 1, "b"), (False, 2, "c"), (True, 1, "d"),
        ])
        examples, counters = self.build(turns)
        assert len(examples) == 2
        assert counters["turns_mine"] == 2

    def test_context_grows_as_the_conversation_proceeds(self):
        turns = conversation([
            (False, 2, "a"), (True, 1, "b"), (False, 2, "c"), (True, 1, "d"),
        ])
        examples, _ = self.build(turns)
        assert len(examples[0]["context"]) == 1
        assert len(examples[1]["context"]) == 3

    def test_trivial_turns_are_downsampled(self):
        turns = [turn("hi", mine=False, sender=2, minute=0)]
        turns += [turn("ok", minute=i) for i in range(1, 60)]
        kept_all, _ = self.build(turns, trivial_keep_rate=1.0)
        kept_none, _ = self.build(turns, trivial_keep_rate=0.0)
        assert len(kept_none) == 0
        assert len(kept_all) > 0

    def test_substantive_turns_are_never_downsampled(self):
        turns = [turn("hi", mine=False, sender=2, minute=0)]
        turns += [turn("something substantive here", minute=i) for i in range(1, 20)]
        examples, counters = self.build(turns, trivial_keep_rate=0.0)
        assert len(examples) == 19
        assert counters["trivial_seen"] == 0

    def test_rebuild_is_identical(self):
        turns = [turn("hi", mine=False, sender=2, minute=0)]
        turns += [turn("ok", minute=i) for i in range(1, 40)]
        first, _ = self.build(turns, trivial_keep_rate=0.33)
        second, _ = self.build(turns, trivial_keep_rate=0.33)
        assert first == second

    def test_examples_carry_no_real_sender_ids(self):
        turns = conversation([(False, 987654321, "hi"), (True, 1, "yo")])
        examples, _ = self.build(turns)
        assert "987654321" not in str(examples[0]["context"])

    def test_sessions_do_not_share_context(self):
        early = [turn("hi", mine=False, sender=2, hour=2), turn("yo", hour=2, minute=1)]
        late = [turn("back", mine=False, sender=2, hour=20), turn("hey", hour=20, minute=1)]
        pseudonyms = assign_pseudonyms(early + late)
        config = BuildConfig(trivial_keep_rate=1.0)
        examples, _ = build_examples(1, "private", [early, late], pseudonyms, config)
        assert len(examples) == 2
        assert [c["text"] for c in examples[1]["context"]] == ["back"]


class TestHoldoutSelection:
    def test_largest_chats_are_never_held_out(self):
        ranked = list(range(20))
        assert not set(select_holdout_chats(ranked, 3)) & {0, 1, 2}

    def test_requested_count_is_returned(self):
        assert len(select_holdout_chats(list(range(50)), 4)) == 4

    def test_selection_is_deterministic(self):
        ranked = list(range(50))
        assert select_holdout_chats(ranked, 4) == select_holdout_chats(ranked, 4)

    def test_no_duplicates(self):
        picked = select_holdout_chats(list(range(50)), 4)
        assert len(set(picked)) == len(picked)

    def test_selection_spans_the_size_range(self):
        picked = select_holdout_chats(list(range(100)), 4)
        assert max(picked) - min(picked) > 40

    def test_zero_requested(self):
        assert select_holdout_chats([1, 2, 3], 0) == []

    def test_more_requested_than_available(self):
        assert len(select_holdout_chats([1, 2, 3, 4], 10)) <= 4

    def test_tiny_pool_falls_back_to_all_chats(self):
        assert select_holdout_chats([1, 2], 1) != []
