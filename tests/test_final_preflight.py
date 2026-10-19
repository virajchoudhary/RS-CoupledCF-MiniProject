"""Focused tests for the fail-closed academic hand-in gate."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from final_preflight import EXPECTED_STUDENTS, validate_roster, validate_final


def write_roster(tmp_path, *, missing=False, verified=False):
    path = tmp_path / "configs"
    path.mkdir()
    students = [
        {"name": name, "roll_number": roll,
         "contribution": ("Implemented or reviewed the documented component and checked the evidence"
                          if verified else "[Actual individual contribution pending student confirmation]")}
        for name, roll in sorted(EXPECTED_STUDENTS)
    ]
    if missing:
        students.pop()
    (path / "student.json").write_text(json.dumps({
        "course": "Recommendation Systems",
        "institution": "SVKM's NMIMS MPSTME, Mumbai",
        "group": "Group 4",
        "students": students,
    }), encoding="utf-8")
    return students


def test_official_roster_accepts_four_members_with_pending_contributions(tmp_path):
    write_roster(tmp_path)
    _, members, pending = validate_roster(tmp_path)
    assert len(members) == 4
    assert len(pending) == 4


def test_roster_rejects_missing_student(tmp_path):
    write_roster(tmp_path, missing=True)
    with pytest.raises(ValueError, match="exactly four"):
        validate_roster(tmp_path)


def test_roster_rejects_invented_assignment(tmp_path):
    students = write_roster(tmp_path)
    students[0]["roll_number"] = "I999"
    (tmp_path / "configs" / "student.json").write_text(json.dumps({
        "course": "Recommendation Systems",
        "institution": "SVKM's NMIMS MPSTME, Mumbai",
        "group": "Group 4",
        "students": students,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        validate_roster(tmp_path)


def test_final_gate_rejects_unconfirmed_contributions_before_export_checks(tmp_path):
    write_roster(tmp_path)
    with pytest.raises(ValueError, match="Contributions require human confirmation"):
        validate_final(tmp_path)


def test_nonplaceholder_contributions_clear_metadata_gate_without_claiming_human_approval(tmp_path):
    write_roster(tmp_path, verified=True)
    _, _, pending = validate_roster(tmp_path)
    assert pending == []
