"""Tests for chat-message conversion.

All sample data is fictional and hand-written.
"""

from mini_gabriel.chatformat import (
    ASSISTANT,
    DEFAULT_SYSTEM_PROMPT,
    ME,
    SYSTEM,
    USER,
    convert_all,
    needs_speaker_prefix,
    to_messages,
)


def example(context, target):
    return {
        "chat_id": 1,
        "chat_type": "private",
        "context": [{"speaker": s, "text": t} for s, t in context],
        "target": target,
    }


class TestRoles:
    def test_other_speaker_becomes_user_and_me_becomes_assistant(self):
        messages = to_messages(example([("A", "you free")], ["ya"]), system_prompt=None)
        assert [m["role"] for m in messages] == [USER, ASSISTANT]

    def test_system_prompt_is_prepended(self):
        messages = to_messages(example([("A", "hi")], ["yo"]))
        assert messages[0]["role"] == SYSTEM
        assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_system_prompt_can_be_omitted(self):
        messages = to_messages(example([("A", "hi")], ["yo"]), system_prompt=None)
        assert all(m["role"] != SYSTEM for m in messages)

    def test_target_is_the_final_assistant_message(self):
        messages = to_messages(example([("A", "hi")], ["ya", "6pm"]), system_prompt=None)
        assert messages[-1] == {"role": ASSISTANT, "content": "ya\n6pm"}

    def test_burst_boundaries_survive_as_newlines(self):
        messages = to_messages(example([("A", "hi")], ["ya", "6pm", "ok"]), system_prompt=None)
        assert messages[-1]["content"].count("\n") == 2


class TestAlternation:
    def test_roles_alternate(self):
        messages = to_messages(
            example([("A", "one"), (ME, "two"), ("A", "three")], ["four"]),
            system_prompt=None,
        )
        assert [m["role"] for m in messages] == [USER, ASSISTANT, USER, ASSISTANT]

    def test_consecutive_users_are_merged(self):
        # Two different people speaking in a row both map to the user role.
        messages = to_messages(
            example([("A", "one"), ("B", "two")], ["three"]), system_prompt=None
        )
        assert [m["role"] for m in messages] == [USER, ASSISTANT]
        assert "one" in messages[0]["content"] and "two" in messages[0]["content"]

    def test_no_two_adjacent_messages_share_a_role(self):
        messages = to_messages(
            example([("A", "a"), ("B", "b"), ("C", "c"), (ME, "d"), ("A", "e")], ["f"]),
            system_prompt=None,
        )
        roles = [m["role"] for m in messages]
        assert all(first != second for first, second in zip(roles, roles[1:]))


class TestSpeakerPrefixes:
    def test_one_to_one_chat_gets_no_prefix(self):
        messages = to_messages(example([("A", "you free")], ["ya"]), system_prompt=None)
        assert messages[0]["content"] == "you free"

    def test_group_chat_labels_who_is_speaking(self):
        messages = to_messages(
            example([("A", "one"), ("B", "two")], ["three"]), system_prompt=None
        )
        assert "A: one" in messages[0]["content"]
        assert "B: two" in messages[0]["content"]

    def test_prefix_detection(self):
        assert not needs_speaker_prefix([{"speaker": "A", "text": "x"}])
        assert needs_speaker_prefix(
            [{"speaker": "A", "text": "x"}, {"speaker": "B", "text": "y"}]
        )

    def test_my_own_turns_are_never_prefixed(self):
        # My turn must sit mid-context: an example whose context ends with my
        # own turn is dropped, so it cannot be used to test prefixing.
        messages = to_messages(
            example([("A", "one"), (ME, "mine"), ("B", "two")], ["reply"]),
            system_prompt=None,
        )
        assert messages[1]["content"] == "mine"
        assert messages[0]["content"] == "A: one"

    def test_prefix_can_be_forced(self):
        messages = to_messages(
            example([("A", "hi")], ["yo"]), system_prompt=None, force_speaker_prefix=True
        )
        assert messages[0]["content"] == "A: hi"


class TestDropped:
    def test_context_of_only_my_messages_is_dropped(self):
        # Nothing incoming to reply to.
        assert to_messages(example([(ME, "earlier")], ["later"])) is None

    def test_empty_context_is_dropped(self):
        assert to_messages(example([], ["hi"])) is None

    def test_empty_target_is_dropped(self):
        assert to_messages(example([("A", "hi")], [])) is None

    def test_target_is_never_merged_into_the_context(self):
        # If the last context turn were mine, merging would train the model to
        # predict its own context rather than the reply.
        result = to_messages(example([("A", "hi"), (ME, "mine")], ["reply"]), system_prompt=None)
        assert result is None


class TestConvertAll:
    def test_counts_conversions_and_drops(self):
        rows, counters = convert_all([
            example([("A", "hi")], ["yo"]),
            example([(ME, "solo")], ["later"]),
            example([("A", "hi"), (ME, "mine")], ["reply"]),
        ])
        assert counters["converted"] == 1
        assert counters["dropped_no_other_speaker"] == 1
        assert counters["dropped_trailing_self"] == 1
        assert len(rows) == 1

    def test_rows_have_the_expected_shape(self):
        rows, _ = convert_all([example([("A", "hi")], ["yo"])])
        assert set(rows[0]) == {"messages"}
        assert all(set(m) == {"role", "content"} for m in rows[0]["messages"])

    def test_empty_input(self):
        rows, counters = convert_all([])
        assert rows == [] and counters["converted"] == 0
