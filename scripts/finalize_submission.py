"""One-command, fail-closed final submission rebuild in the original local workspace.

Requires the official MovieLens data, selected checkpoints, PDF dependencies,
and the original Codex presentation runtime. Never reruns the 15 full training
runs or silently generates substitute metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from final_preflight import validate_roster, validate_final

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*args: str) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and regenerate the Group 4 hand-in")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pytest only; experimental and artifact verification still run")
    parser.add_argument("--preview", action="store_true",
                        help="Regenerate report, slides and evidence before contributions are confirmed; do not package")
    args = parser.parse_args()
    _, members, pending = validate_roster()
    if pending and not args.preview:
        raise SystemExit(
            "STOP: the following members must confirm their actual individual contributions "
            "in configs/student.json: " + ", ".join(pending)
        )
    print("Roster is verified and ordered. Planned speaking roles are not claims of completed work.")
    if pending:
        print("PREVIEW ONLY: actual individual contributions pending from " + ", ".join(pending))
    if not args.skip_tests:
        run(PYTHON, "-m", "pytest", "-q")
    run(PYTHON, "scripts/verify_submission.py", "--experiments-only")
    run(PYTHON, "scripts/build_supporting_docs.py")
    run(PYTHON, "scripts/build_report.py")
    run(PYTHON, "scripts/build_notebook.py")
    run(PYTHON, "scripts/execute_notebook.py")
    run("node", "scripts/build_presentation.mjs", "--final")
    run(PYTHON, "scripts/check_documents.py")
    run(PYTHON, "scripts/verify_submission.py")
    shutil.copyfile(ROOT / ".audit/verification.json",
                    ROOT / "deliverables/verification.json")
    if args.preview:
        print("PREVIEW PASS: regenerated artifacts, but final ZIP is deliberately blocked "
              "until the students confirm real contributions and approve the exports.")
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
