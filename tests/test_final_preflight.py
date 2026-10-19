"""Focused tests for the fail-closed academic hand-in gate."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from final_preflight import EXPECTED_ROLL_ORDER, validate_roster, validate_final


def write_roster(tmp_path, *, missing=False, verified=False, reverse=False, confirmed=False):
    path = tmp_path / "configs"
    path.mkdir()
    members = [
        ("Dhruva Ambhaikar", "I002", "1-3", "Problem and research gap"),
        ("Viraj Choudhary", "I012", "4-7", "Architecture and implementation"),
        ("Rusheel Sharma", "I069", "8-12", "Data, metrics and experiments"),
        ("Shawn Mendes", "I077", "13-16", "Analysis and conclusions"),
    ]
    students = [
        {
            "name": name,
            "roll_number": roll,
            "contribution": ("Reviewed the project experiment results and prepared the assigned presentation"
                             if verified else "[Actual individual contribution pending student confirmation]"),
            "contribution_confirmed": confirmed,
            "presentation_slides": slides,
            "presentation_responsibility": "Assigned presentation focus: " + short,
            "presentation_short": short,
        }
        for name, roll, slides, short in members
    ]
    if missing:
        students.pop()
    if reverse:
        students.reverse()
    (path / "student.json").write_text(json.dumps({
        "course": "Recommendation Systems",
        "institution": "SVKM's NMIMS MPSTME, Mumbai",
        "group": "Group 4",
        "students": students,
    }), encoding="utf-8")
    return students


def test_official_roster_sorted_and_pending(tmp_path):
    write_roster(tmp_path)
    _, members, pending = validate_roster(tmp_path)
    assert tuple(m["roll_number"] for m in members) == EXPECTED_ROLL_ORDER
    assert len(pending) == 4


def test_roster_rejects_missing_student(tmp_path):
    write_roster(tmp_path, missing=True)
    with pytest.raises(ValueError, match="exactly four"):
        validate_roster(tmp_path)


def test_roster_rejects_wrong_roll_number(tmp_path):
    students = write_roster(tmp_path)
    students[0]["roll_number"] = "I999"
    (tmp_path / "configs" / "student.json").write_text(json.dumps({
        "course": "Recommendation Systems", "institution": "SVKM's NMIMS MPSTME, Mumbai",
        "group": "Group 4", "students": students,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        validate_roster(tmp_path)


def test_roster_rejects_wrong_order(tmp_path):
    write_roster(tmp_path, reverse=True)
    with pytest.raises(ValueError, match="ascending roll-number order"):
        validate_roster(tmp_path)


def test_roster_rejects_unassigned_slides(tmp_path):
    students = write_roster(tmp_path)
    students[1]["presentation_slides"] = "8-12"
    (tmp_path / "configs" / "student.json").write_text(json.dumps({
        "course": "Recommendation Systems", "institution": "SVKM's NMIMS MPSTME, Mumbai",
        "group": "Group 4", "students": students,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="slide range"):
        validate_roster(tmp_path)


def test_final_gate_rejects_unconfirmed_contributions_before_export_checks(tmp_path):
    write_roster(tmp_path)
    with pytest.raises(ValueError, match="Contributions require human confirmation"):
        validate_final(tmp_path)


def test_contribution_text_alone_does_not_confirm_academic_work(tmp_path):
    write_roster(tmp_path, verified=True)
    _, _, pending = validate_roster(tmp_path)
    assert len(pending) == 4


def test_explicit_confirmation_and_real_text_clear_metadata_gate(tmp_path):
    write_roster(tmp_path, verified=True, confirmed=True)
    _, _, pending = validate_roster(tmp_path)
    assert pending == []


def test_confirmation_does_not_accept_placeholder_contributions(tmp_path):
    write_roster(tmp_path, confirmed=True)
    _, _, pending = validate_roster(tmp_path)
    assert len(pending) == 4
