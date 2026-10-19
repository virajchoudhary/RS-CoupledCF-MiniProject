"""Preserve inherited artifacts while rebuilding a non-submittable preview."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finalize_submission


def test_preview_archives_stale_package_without_touching_research(tmp_path):
    output = tmp_path / "deliverables"
    output.mkdir()
    for name in ("submission.zip", "submission.sha256", "artifact_manifest.json"):
        (output / name).write_bytes(name.encode())
    checkpoint = tmp_path / "results/main/coupled_seed42/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"protected checkpoint")
    backup = finalize_submission.archive_package(tmp_path)
    for name in ("submission.zip", "submission.sha256", "artifact_manifest.json"):
        assert not (output / name).exists()
        assert (backup / name).read_bytes() == name.encode()
    assert checkpoint.read_bytes() == b"protected checkpoint"
    assert finalize_submission.archive_package(tmp_path) is None


def test_visual_review_must_match_current_exports(tmp_path):
    import hashlib
    import json
    from final_preflight import validate_visual_review

    (tmp_path / "deliverables").mkdir()
    (tmp_path / ".audit").mkdir()
    pdf = tmp_path / "deliverables/final_report.pdf"
    pptx = tmp_path / "deliverables/CoupledCF_presentation.pptx"
    pdf.write_bytes(b"reviewed PDF")
    pptx.write_bytes(b"reviewed deck")
    receipt = {
        "status": "PASS",
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pptx_sha256": hashlib.sha256(pptx.read_bytes()).hexdigest(),
    }
    (tmp_path / ".audit/visual_qa.json").write_text(json.dumps(receipt))
    validate_visual_review(tmp_path)
    pdf.write_bytes(b"changed after inspection")
    import pytest
    with pytest.raises(ValueError, match="stale"):
        validate_visual_review(tmp_path)


def test_failed_zip_build_preserves_existing_archive(tmp_path):
    from package_submission import write_archive
    import pytest

    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous complete archive")
    with pytest.raises(FileNotFoundError):
        write_archive(tmp_path, [tmp_path / "missing.py"], {}, output)
    assert output.read_bytes() == b"previous complete archive"
    assert list(tmp_path.glob("*.tmp")) == []


def test_swapped_slide_topics_reject_incorrect_speaking_ranges(tmp_path):
    import zipfile
    from final_preflight import SLIDE_TOPICS, _slide_text
    import pytest

    deck = tmp_path / "swapped.pptx"
    topics = list(SLIDE_TOPICS)
    topics[2], topics[3] = topics[3], topics[2]
    with zipfile.ZipFile(deck, "w") as archive:
        for number, topic in enumerate(topics, 1):
            archive.writestr(
                f"ppt/slides/slide{number}.xml",
                '<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f'<a:r><a:t>{topic}</a:t></a:r></a:p>',
            )
    with pytest.raises(ValueError, match="Slide 3 topic"):
        _slide_text(deck)
