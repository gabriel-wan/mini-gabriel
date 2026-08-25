"""Tests for the analysis and reporting stage.

All sample data is fictional. The most important test here is
``test_message_text_never_reaches_the_report``: the report is the one artefact
that summarises private conversations, and it must never carry their contents.
"""

import json

from mini_gabriel.analyze import (
    analyse_chats,
    build_report,
    render_markdown,
    render_terminal_summary,
    write_report,
)
from mini_gabriel.selection import SelectionCriteria
from mini_gabriel.storage import (
    append_records,
    chat_jsonl_path,
    empty_manifest,
    manifest_chat_entry,
)

WINDOW_START = "2025-12-31T16:00:00+00:00"
WINDOW_END = "2026-12-31T16:00:00+00:00"


def make_manifest():
    return empty_manifest(2026, "Asia/Singapore", WINDOW_START, WINDOW_END)


def add_chat(manifest, chats_dir, chat_id, name, chat_type, participants, my_text, other_text=5,
             participant_count_known=True, complete=True, text="a fictional message"):
    """Add a fictional chat plus its raw messages."""
    records = []
    for index in range(my_text):
        records.append({
            "chat_id": chat_id,
            "message_id": index + 1,
            "date_utc": "2026-03-01T04:00:00+00:00",
            "is_outgoing": True,
            "text": text,
        })
    for index in range(other_text):
        records.append({
            "chat_id": chat_id,
            "message_id": my_text + index + 1,
            "date_utc": "2026-03-02T04:00:00+00:00",
            "is_outgoing": False,
            "text": "a fictional reply",
        })
    append_records(chat_jsonl_path(chats_dir, chat_id), records)

    manifest["chats"][str(chat_id)] = manifest_chat_entry(
        {
            "chat_id": chat_id,
            "name": name,
            "chat_type": chat_type,
            "participant_count": participants,
            "participant_count_known": participant_count_known,
            "is_bot": False,
            "is_self_chat": False,
        },
        last_id=len(records),
        message_count=len(records),
        complete=complete,
    )
    return manifest


def build_fixture(tmp_path):
    chats_dir = tmp_path / "chats"
    manifest = make_manifest()
    add_chat(manifest, chats_dir, 1001, "Fictional Friend", "private", 2, my_text=150)
    add_chat(manifest, chats_dir, 1002, "Fictional Acquaintance", "private", 2, my_text=12)
    add_chat(manifest, chats_dir, -2001, "Fictional Small Group", "supergroup", 6, my_text=120)
    add_chat(manifest, chats_dir, -2002, "Fictional Big Group", "supergroup", 250, my_text=400)
    add_chat(
        manifest, chats_dir, -2003, "Fictional Opaque Group", "supergroup", None,
        my_text=300, participant_count_known=False,
    )
    manifest["skipped"]["-3001"] = {
        "chat_id": -3001,
        "name": "Fictional News Channel",
        "chat_type": "broadcast",
        "reason": "broadcast channel",
    }
    return manifest, chats_dir


class TestAnalysis:
    def test_qualifying_chats_are_identified(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))

        qualifying = {row["name"] for row in report["chats"] if row["qualifies"]}
        assert qualifying == {"Fictional Friend", "Fictional Small Group"}

    def test_rejection_reasons_are_specific(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        by_name = {row["name"]: row for row in report["chats"]}

        assert "exceeds limit" in " ".join(by_name["Fictional Big Group"]["reasons"])
        assert "could not be determined" in " ".join(by_name["Fictional Opaque Group"]["reasons"])
        assert "need 100" in " ".join(by_name["Fictional Acquaintance"]["reasons"])

    def test_results_are_sorted_by_my_text_volume(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        results = analyse_chats(manifest, chats_dir)
        counts = [stats.my_text_messages for stats, _ in results]
        assert counts == sorted(counts, reverse=True)

    def test_totals_add_up(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))

        assert report["totals"]["chats_extracted"] == 5
        assert report["totals"]["chats_qualifying"] == 2
        assert report["totals"]["dialogs_skipped"] == 1
        assert report["totals"]["my_text_messages_total"] == 150 + 12 + 120 + 400 + 300
        assert report["totals"]["my_text_messages_in_qualifying_chats"] == 150 + 120

    def test_window_metadata_is_carried_through(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        assert report["window"]["timezone"] == "Asia/Singapore"
        assert report["window"]["window_start_utc"] == WINDOW_START

    def test_looser_criteria_admit_more_chats(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        loose = SelectionCriteria(max_participants=1000, min_my_text_messages=10)
        report = build_report(manifest, analyse_chats(manifest, chats_dir, loose), loose)
        # The opaque group still fails: an unknown count is never assumed small.
        assert report["totals"]["chats_qualifying"] == 4

    def test_empty_manifest_produces_an_empty_report(self, tmp_path):
        manifest = make_manifest()
        report = build_report(manifest, analyse_chats(manifest, tmp_path / "chats"))
        assert report["totals"]["chats_extracted"] == 0
        assert report["chats"] == []


class TestPrivacy:
    def test_message_text_never_reaches_the_report(self, tmp_path):
        """The report summarises private chats; it must carry no message text."""
        secret = "FICTIONAL-CANARY-PHRASE-9f3a2b"
        chats_dir = tmp_path / "chats"
        manifest = make_manifest()
        add_chat(manifest, chats_dir, 1001, "Fictional Friend", "private", 2,
                 my_text=150, text=secret)

        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        rendered = "".join([
            json.dumps(report, ensure_ascii=False),
            render_markdown(report),
            render_terminal_summary(report),
        ])
        assert secret not in rendered

    def test_written_report_files_contain_no_message_text(self, tmp_path):
        secret = "FICTIONAL-CANARY-PHRASE-9f3a2b"
        chats_dir = tmp_path / "chats"
        manifest = make_manifest()
        add_chat(manifest, chats_dir, 1001, "Fictional Friend", "private", 2,
                 my_text=150, text=secret)

        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        json_path, markdown_path = write_report(report, tmp_path / "processed")

        assert secret not in json_path.read_text(encoding="utf-8")
        assert secret not in markdown_path.read_text(encoding="utf-8")


class TestRendering:
    def test_markdown_lists_every_chat(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        markdown = render_markdown(build_report(manifest, analyse_chats(manifest, chats_dir)))
        for name in ("Fictional Friend", "Fictional Small Group", "Fictional Big Group"):
            assert name in markdown

    def test_markdown_reports_skipped_dialogs(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        markdown = render_markdown(build_report(manifest, analyse_chats(manifest, chats_dir)))
        assert "Fictional News Channel" in markdown
        assert "broadcast channel" in markdown

    def test_pipe_in_chat_name_cannot_break_the_table(self, tmp_path):
        chats_dir = tmp_path / "chats"
        manifest = make_manifest()
        add_chat(manifest, chats_dir, 1001, "Weird | Name", "private", 2, my_text=150)
        markdown = render_markdown(build_report(manifest, analyse_chats(manifest, chats_dir)))
        assert "Weird \\| Name" in markdown

    def test_unknown_participant_count_renders_as_unknown(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        assert "unknown" in render_markdown(report)

    def test_terminal_summary_truncates_long_names(self, tmp_path):
        chats_dir = tmp_path / "chats"
        manifest = make_manifest()
        add_chat(manifest, chats_dir, 1001, "F" * 80, "private", 2, my_text=150)
        summary = render_terminal_summary(build_report(manifest, analyse_chats(manifest, chats_dir)))
        assert "..." in summary
        assert "F" * 80 not in summary

    def test_write_report_creates_both_files(self, tmp_path):
        manifest, chats_dir = build_fixture(tmp_path)
        report = build_report(manifest, analyse_chats(manifest, chats_dir))
        json_path, markdown_path = write_report(report, tmp_path / "processed")
        assert json_path.exists() and markdown_path.exists()
        assert json.loads(json_path.read_text(encoding="utf-8"))["totals"]["chats_extracted"] == 5
