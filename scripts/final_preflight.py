"""Fail-closed final hand-in gate for student attribution and exported artifacts.

Run python scripts/final_preflight.py --roster-only before the students provide
their real contributions. Run it without arguments before packaging.
This gate does not rerun training or independently certify the experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GROUP = "Group 4"
EXPECTED_STUDENTS = {
    ("Dhruva Ambhaikar", "I002"),
    ("Viraj Choudhary", "I012"),
    ("Rusheel Sharma", "I069"),
    ("Shawn Mendes", "I077"),
}
EXPECTED_ROLL_ORDER = ("I002", "I012", "I069", "I077")
EXPECTED_SLIDE_SEGMENTS = {
    "I002": "1-3",
    "I012": "4-7",
    "I069": "8-12",
    "I077": "13-16",
}
PLACEHOLDERS = (
    "[Student name]",
    "[Roll number]",
    "[Describe actual individual contribution]",
    "[Actual individual contribution pending student confirmation]",
    "[Institution]",
    "[Group number]",
)
NAMESPACE = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_roster(root: Path = ROOT) -> tuple[dict, list[dict], list[str]]:
    """Check group allocation exactly; return unconfirmed student names."""
    config = json.loads((root / "configs/student.json").read_text(encoding="utf-8"))
    _require(config.get("group") == EXPECTED_GROUP, "Wrong or missing Group 4 allocation")
    _require(config.get("course") == "Recommendation Systems", "Course name is missing or changed")
    _require(bool(str(config.get("institution", "")).strip()) and
             "[" not in str(config.get("institution", "")), "Institution is incomplete")
    members = config.get("students")
    _require(isinstance(members, list) and len(members) == len(EXPECTED_STUDENTS),
             "Expected exactly four assigned Group 4 students")
    actual = [(member.get("name"), member.get("roll_number")) for member in members]
    _require(set(actual) == EXPECTED_STUDENTS and len(actual) == len(set(actual)),
             "Student identities do not match the supplied official RS.xlsx Group 4 allocation")
    _require(tuple(member["roll_number"] for member in members) == EXPECTED_ROLL_ORDER,
             "Student roster must be in ascending roll-number order: I002, I012, I069, I077")
    for member in members:
        _require(member.get("presentation_slides") ==
                 EXPECTED_SLIDE_SEGMENTS[member["roll_number"]],
                 f"Missing or inaccurate assigned slide range for {member['name']}")
        _require(bool(str(member.get("presentation_responsibility", "")).strip()) and
                 bool(str(member.get("presentation_short", "")).strip()),
                 f"Missing non-credit viva responsibilities for {member['name']}")
    pending = []
    for member in members:
        contribution = str(member.get("contribution", "")).strip()
        if (not contribution or "[" in contribution or "]" in contribution or
                "pending" in contribution.casefold() or
                "to be confirmed" in contribution.casefold() or len(contribution) < 12):
            pending.append(member["name"])
    return config, members, pending


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install requirements-artifacts.txt to validate the exported PDF") from exc
    reader = PdfReader(str(path))
    _require(len(reader.pages) >= 8, "Final academic PDF is unexpectedly short")
    return _normal(" ".join(page.extract_text() or "" for page in reader.pages))


def _slide_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            (name for name in archive.namelist()
             if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
        )
        _require(len(slides) == 16, "Expected the verified 16-slide deck")
        text = []
        for name in slides:
            document = ET.fromstring(archive.read(name))
            text.extend((element.text or "") for element in document.findall(".//a:t", NAMESPACE))
    return _normal(" ".join(text))


def validate_exports(root: Path, members: list[dict]) -> None:
    """Reject placeholders, missing roster entries, or stale experiment numbers."""
    pdf = _pdf_text(root / "deliverables/final_report.pdf")
    slides = _slide_text(root / "deliverables/CoupledCF_presentation.pptx")
    for source_name, source in (("PDF", pdf), ("PowerPoint", slides)):
        indices = []
        for member in members:
            _require(member["name"] in source and member["roll_number"] in source,
                     f"{source_name} missing {member['name']} ({member['roll_number']}); regenerate it")
            indices.append(source.index(member["roll_number"]))
            if source_name == "PDF":
                _require(_normal(member["contribution"]) in source,
                         f"PDF contribution for {member['name']} is missing or stale")
            else:
                _require(member["presentation_short"] in source and
                         member["presentation_slides"] in source,
                         f"PowerPoint assigned viva role for {member['name']} is missing or stale")
        _require(indices == sorted(indices), f"{source_name} roster is not in roll-number order")
        for placeholder in PLACEHOLDERS:
            _require(placeholder not in source,
                     f"{source_name} still contains placeholder {placeholder!r}")
    with (root / "results/summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 5, "Expected five experimental variants in results/summary.csv")
    for row in rows:
        for metric in ("HR@10_mean", "NDCG@10_mean"):
            formatted = f"{float(row[metric]):.4f}"
            _require(formatted in pdf and formatted in slides,
                     f"Final exports are missing recorded {row['variant']} {metric}: {formatted}")
    receipt_path = root / "deliverables/verification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("status") == "PASS" and len(receipt.get("runs", [])) == 15,
             "The recorded 15-run experimental verification is absent or incomplete")


def validate_final(root: Path = ROOT) -> None:
    """Gate used by the ZIP packager; does not substitute for experiment replay."""
    _, members, pending = validate_roster(root)
    _require(not pending, "Contributions require human confirmation: " + ", ".join(pending))
    validate_exports(root, members)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group 4 final academic hand-in preflight")
    parser.add_argument("--roster-only", action="store_true",
                        help="Verify identities without requiring contributions or regenerated exports")
    args = parser.parse_args()
    _, _, pending = validate_roster()
    if args.roster_only:
        print("PASS: official four-person Group 4 roster matches the assignment spreadsheet")
        if pending:
            print("HUMAN INPUT STILL REQUIRED: verify individual contributions for " + ", ".join(pending))
        else:
            print("All four contribution fields have non-placeholder text; students must confirm accuracy")
        return
    validate_final()
    print("PASS: identities, non-placeholder contributions, exports, metrics, and recorded 15-run receipt")
    print("Human responsibility remains: verify the truth of each contribution and approve the hand-in")


if __name__ == "__main__":
    main()
