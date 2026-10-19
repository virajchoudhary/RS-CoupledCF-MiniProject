"""Data-independent guardrails for prepared academic hand-in materials."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("I002", "I012", "I069", "I077")


def test_required_presentation_files_exist():
    for name in (
        "docs/GROUP4_VIVA_PLAYBOOK.md",
        "docs/VIVA_LAST_MINUTE.md",
        "docs/FINAL_HANDOFF.md",
        "docs/final_report.md",
        "deliverables/submission_guide.md",
    ):
        assert (ROOT / name).is_file(), name


def test_all_public_group_docs_use_ascending_roll_number_order():
    for name in (
        "docs/GROUP4_VIVA_PLAYBOOK.md",
        "docs/VIVA_LAST_MINUTE.md",
        "docs/FINAL_HANDOFF.md",
        "docs/final_report.md",
        "deliverables/submission_guide.md",
        "docs/requirements_checklist.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        positions = [text.find(roll) for roll in ORDER]
        assert all(pos >= 0 for pos in positions), name
        assert positions == sorted(positions), f"{name}: group must appear in roll order"


def test_viva_material_matches_measured_scores_and_preserves_limits():
    with (ROOT / "results/summary.csv").open(newline="", encoding="utf-8") as fp:
        results = {row["variant"]: row for row in csv.DictReader(fp)}
    assert set(results) == {"deepcf", "local", "global", "coupled_local", "coupled"}
    guide = (ROOT / "docs/GROUP4_VIVA_PLAYBOOK.md").read_text(encoding="utf-8")
    short = (ROOT / "docs/VIVA_LAST_MINUTE.md").read_text(encoding="utf-8")
    for model in results.values():
        for metric in ("HR@10_mean", "NDCG@10_mean"):
            number = f"{float(model[metric]):.4f}"
            assert number in guide, f"long guide omitted measured {number}"
            assert number in short, f"revision sheet omitted measured {number}"
    assert "not" in short.lower() and "cold-start" in short.lower()
    assert "partial reproduction" in guide.lower()
    assert "statistical significance" in guide.lower()


def test_handoff_does_not_treat_assigned_speaking_as_proven_research_work():
    for name in ("docs/GROUP4_VIVA_PLAYBOOK.md", "docs/FINAL_HANDOFF.md"):
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "not" in text and "actual" in text
    roster = (ROOT / "configs/student.json").read_text(encoding="utf-8")
    assert '"presentation_responsibility"' in roster
    assert '"contribution"' in roster
