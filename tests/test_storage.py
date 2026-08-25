"""Tests for JSONL and manifest storage.

All sample data is fictional. Tests write only into pytest's tmp_path.
"""

import json

from mini_gabriel.storage import (
    append_records,
    chat_jsonl_path,
    count_records,
    empty_manifest,
    iter_records,
    last_message_id,
    load_manifest,
    manifest_chat_entry,
    save_manifest,
)


def make_record(message_id, text="a fictional message"):
    return {
        "chat_id": 1001,
        "message_id": message_id,
        "date_utc": "2026-03-01T04:00:00+00:00",
        "is_outgoing": True,
        "text": text,
    }


class TestChatPaths:
    def test_positive_id(self, tmp_path):
        assert chat_jsonl_path(tmp_path, 1234).name == "1234.jsonl"

    def test_negative_id_is_encoded_without_a_minus_sign(self, tmp_path):
        # Group and channel ids are negative; a leading '-' is awkward in
        # filenames and shell arguments.
        assert chat_jsonl_path(tmp_path, -1234).name == "n1234.jsonl"


class TestJsonlRoundTrip:
    def test_append_then_read(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        written = append_records(path, [make_record(1), make_record(2)])
        assert written == 2
        assert [r["message_id"] for r in iter_records(path)] == [1, 2]

    def test_appending_preserves_earlier_records(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(1)])
        append_records(path, [make_record(2)])
        assert count_records(path) == 2

    def test_text_is_preserved_exactly(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        awkward = "  spaced  \n newline \t tab  emoji-safe: ✨  "
        append_records(path, [make_record(1, text=awkward)])
        assert next(iter_records(path))["text"] == awkward

    def test_non_ascii_is_not_escaped_away(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(1, text="你好 lah")])
        assert next(iter_records(path))["text"] == "你好 lah"

    def test_missing_file_yields_nothing(self, tmp_path):
        assert list(iter_records(tmp_path / "absent.jsonl")) == []

    def test_truncated_final_line_is_skipped(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(1)])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"message_id": 2, "text": "cut off')
        assert [r["message_id"] for r in iter_records(path)] == [1]

    def test_blank_lines_are_ignored(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(1)])
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert count_records(path) == 1


class TestResumePoint:
    def test_highest_id_is_the_resume_point(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(5), make_record(9), make_record(7)])
        assert last_message_id(path) == 9

    def test_empty_chat_resumes_from_zero(self, tmp_path):
        assert last_message_id(tmp_path / "absent.jsonl") == 0

    def test_resume_point_survives_a_truncated_write(self, tmp_path):
        path = chat_jsonl_path(tmp_path, 1001)
        append_records(path, [make_record(3)])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"message_id": 4, "tex')
        assert last_message_id(path) == 3


class TestManifest:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest = empty_manifest(
            2026, "Asia/Singapore", "2025-12-31T16:00:00+00:00", "2026-12-31T16:00:00+00:00"
        )
        manifest["chats"]["1001"] = manifest_chat_entry(
            {"chat_id": 1001, "name": "Fictional Chat", "chat_type": "private"},
            last_id=99,
            message_count=12,
            complete=True,
        )
        save_manifest(path, manifest)

        loaded = load_manifest(path)
        assert loaded["target_year"] == 2026
        assert loaded["timezone"] == "Asia/Singapore"
        assert loaded["chats"]["1001"]["last_message_id"] == 99
        assert loaded["chats"]["1001"]["extraction_complete"] is True

    def test_missing_manifest_returns_none(self, tmp_path):
        assert load_manifest(tmp_path / "absent.json") is None

    def test_corrupt_manifest_returns_none_rather_than_raising(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_manifest(path) is None

    def test_save_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "manifest.json"
        save_manifest(path, empty_manifest(2026, "Asia/Singapore", "a", "b"))
        save_manifest(path, empty_manifest(2026, "Asia/Singapore", "a", "b"))
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".manifest-")]
        assert leftovers == []

    def test_entry_records_error_and_incompleteness(self, tmp_path):
        entry = manifest_chat_entry(
            {"chat_id": -7, "name": "Fictional Group", "chat_type": "supergroup"},
            last_id=0,
            message_count=0,
            complete=False,
            error="ChannelPrivateError: no access",
        )
        assert entry["extraction_complete"] is False
        assert "ChannelPrivateError" in entry["error"]

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        path = tmp_path / "manifest.json"
        save_manifest(path, empty_manifest(2026, "Asia/Singapore", "a", "b"))
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
