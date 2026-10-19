"""One-command, fail-closed final submission rebuild in the original local workspace.

Requires the official MovieLens data, selected checkpoints, PDF dependencies,
and the original Codex presentation runtime. Never reruns the 15 full training
runs or silently generates substitute metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from final_preflight import (validate_roster, validate_final, validate_preview,
                             validate_visual_review)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, env=env)


def archive_package(root: Path) -> Path | None:
    files = [root / "deliverables" / name for name in
             ("submission.zip", "submission.sha256", "artifact_manifest.json")]
    existing = [path for path in files if path.exists()]
    if not existing:
        return None
    backup = root / ".audit/submission-history" / uuid4().hex
    backup.mkdir(parents=True)
    for path in existing:
        path.replace(backup / path.name)
    return backup


def record_preview(pending: list[str], *, visually_verified: bool) -> None:
    files = [
        "configs/student.json", "configs/main.json",
        "deliverables/final_report.pdf", "deliverables/CoupledCF_presentation.pptx",
        "deliverables/verification.json", "notebooks/CoupledCF_reproduction.ipynb",
        "notebooks/CoupledCF_reproduction.executed.ipynb", "docs/final_report.md",
        "docs/requirements_checklist.md", "docs/viva_questions.md",
        "docs/GROUP4_VIVA_PLAYBOOK.md", "docs/VIVA_LAST_MINUTE.md",
        "README.md", "docs/methodology.md", "docs/FINAL_HANDOFF.md",
        "deliverables/submission_guide.md",
    ]
    receipt = {
        "status": "PREVIEW_VERIFIED" if visually_verified else "PREVIEW_AWAITING_VISUAL_REVIEW",
        "submission_ready": False,
        "pending_contribution_confirmation": pending,
        "visual_review": "PASS" if visually_verified else "PENDING",
        "artifacts": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in files},
    }
    (ROOT / "deliverables/preview_status.json").write_text(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and regenerate the Group 4 hand-in")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pytest only; experimental and artifact verification still run")
    parser.add_argument("--preview", action="store_true",
                        help="Regenerate report, slides and evidence before contributions are confirmed; do not package")
    parser.add_argument("--verify-preview", action="store_true",
                        help="Verify already rebuilt and visually inspected preview exports without rebuilding")
    parser.add_argument("--package-only", action="store_true",
                        help="Package already rebuilt, confirmed and visually inspected exports without changing them")
    args = parser.parse_args()
    if sum((args.preview, args.verify_preview, args.package_only)) > 1:
        parser.error("Choose only one of --preview, --verify-preview or --package-only")
    _, members, pending = validate_roster()
    if pending and not (args.preview or args.verify_preview):
        raise SystemExit(
            "STOP: the following members must confirm their actual individual contributions "
            "in configs/student.json: " + ", ".join(pending)
        )
    print("Roster is verified and ordered. Planned speaking roles are not claims of completed work.")
    if pending:
        print("PREVIEW ONLY: actual individual contributions pending from " + ", ".join(pending))
    if args.verify_preview:
        run(PYTHON, "scripts/verify_submission.py", "--reuse-experiment-verification")
        shutil.copyfile(ROOT / ".audit/verification.json",
                        ROOT / "deliverables/verification.json")
        validate_preview(ROOT)
        validate_visual_review(ROOT)
        run(PYTHON, "scripts/build_supporting_docs.py")
        record_preview(pending, visually_verified=True)
        print("PREVIEW VERIFIED: current exports inspected; final submission remains a human gate.")
        return
    if args.package_only:
        run(PYTHON, "scripts/verify_submission.py", "--reuse-experiment-verification")
        shutil.copyfile(ROOT / ".audit/verification.json",
                        ROOT / "deliverables/verification.json")
        validate_final(ROOT)
        run(PYTHON, "scripts/package_submission.py")
        print("FINAL PACKAGE PASS: confirmed and reviewed exports packaged without rebuilding.")
        return
    backup = archive_package(ROOT)
    if backup:
        print("Preserved superseded package artifacts:", backup)
    missing = [name for name in ("reportlab", "pypdf")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Install requirements-artifacts.txt into this Python environment: "
                         + ", ".join(missing))
    if not args.skip_tests:
        run(PYTHON, "-m", "pytest", "-q")
    run(PYTHON, "scripts/verify_submission.py", "--experiments-only")
    run(PYTHON, "scripts/build_supporting_docs.py", "--preview")
    run(PYTHON, "scripts/build_report.py")
    run(PYTHON, "scripts/build_notebook.py")
    run(PYTHON, "scripts/execute_notebook.py")
    build_dir = ROOT / ".audit/presentation" / ("rebuild-" + uuid4().hex)
    build_dir.mkdir(parents=True)
    staged = build_dir / "validated.pptx"
    run("node", "scripts/build_presentation.mjs", "--final",
        env={**os.environ, "COUPLEDCF_PRESENTATION_OUTPUT": str(staged)})
    presentation = ROOT / "deliverables/CoupledCF_presentation.pptx"
    if presentation.exists():
        shutil.copy2(presentation, build_dir / "previous.pptx")
    staged.replace(presentation)
    run(PYTHON, "scripts/check_documents.py")
    run(PYTHON, "scripts/verify_submission.py", "--reuse-experiment-verification")
    shutil.copyfile(ROOT / ".audit/verification.json",
                    ROOT / "deliverables/verification.json")
    if args.preview:
        validate_preview(ROOT)
        record_preview(pending, visually_verified=False)
        print("PREVIEW BUILD PASS: inspect the current PDF and every slide, then run "
              "--verify-preview. No final ZIP exists; contributions remain a human gate.")
        return
    validate_final(ROOT)
    run(PYTHON, "scripts/package_submission.py")
    archive = ROOT / "deliverables/submission.zip"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print("FINALIZATION PASS (subject to human authenticity and visual review)")
    print(f"Members: {len(members)}; archive: {archive}; SHA256: {digest}")
    print("Keep the local checkpoints and licensed dataset outside the public submission ZIP.")


if __name__ == "__main__":
    main()
