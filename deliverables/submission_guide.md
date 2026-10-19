# Group 4 submission guide

The assigned paper is **A Generic Framework for Learning Explicit and Implicit User-Item Couplings in Recommendation** (Zhang et al., IEEE Access, 2019). This is a **research-paper study and partial reproduction**, not an exact replication of every experiment.

## Verified roster and planned presentation order

| Roll number | Student | Slides | Assigned preparation responsibility |
| --- | --- | --- | --- |
| I002 | Dhruva Ambhaikar | 1–3 | Research motivation, problem and gap |
| I012 | Viraj Choudhary | 4–7 | CoupledCF architecture, equations and implementation |
| I069 | Rusheel Sharma | 8–12 | Dataset, splits, training, metrics and experiments |
| I077 | Shawn Mendes | 13–16 | Results, paper comparison, ablations, limitations and conclusion |

These are **future speaking and viva assignments**, not assertions about actual prior coding, experimentation, document authorship or individual academic contribution. Every student must confirm their *actual* work in `configs/student.json`.

## Fast preparation build before human contribution confirmation

Run in the **original local workspace**, which retains the licensed MovieLens data and 15 selected checkpoints. The public GitHub repository does not contain these files.

```powershell
git pull origin main
.venv\Scripts\python.exe scripts\final_preflight.py --roster-only
.venv\Scripts\python.exe scripts\finalize_submission.py --preview
```

Preview mode runs unit checks and experiment verification, rebuilds supporting documentation, report and editable presentation, executes the notebook, and checks exported document consistency. It **does not** produce a final submission ZIP and does not treat planned speaking roles as actual academic contributions.

The presenter study guide is at `docs/GROUP4_VIVA_PLAYBOOK.md`. Each student should rehearse their assigned slides and be ready for questions from the other sections too.

## Final build after authentic contributions are confirmed

Each member verifies the work they *actually* performed or reviewed, and the group adds accurate non-placeholder `contribution` statements to `configs/student.json`. Then run:

```powershell
.venv\Scripts\python.exe scripts\finalize_submission.py
```

Final mode runs tests, rechecks the existing experimental evidence, generates PDF/PPTX/notebook, validates the four-person order and attribution, cross-checks the metric means, writes the verification receipt and rebuilds `deliverables/submission.zip`. It does **not** retrain the 15 completed experiments. If the tool stops, investigate the exact blocker and do not claim submission readiness.

## Human approval and actual hand-in

- Confirm actual individual work attribution; do not assign credit based only on who is speaking.
- Examine PDF pages and rendered PowerPoint slides for readability, text overflow, plots, references and correct names/roll numbers.
- Confirm the department's submission format, deadline and any declarations, then personally submit through the required platform.

Deliverables: `deliverables/final_report.pdf`, `deliverables/CoupledCF_presentation.pptx`, `deliverables/submission.zip`, source scripts/notebook, requirements, README, tables and plots.

The ZIP intentionally excludes original MovieLens data, prepared arrays and selected `.pt` checkpoints because the original dataset has redistribution terms and checkpoints can contain derived feature buffers. The original local workspace keeps those for live demonstration; a reviewer can follow the README to rebuild from the official dataset.

The reported results are from five variants on one fixed MovieLens 1M split, three model seeds, ten CPU epochs, and 99 ranking negatives per user. The paper's original two-layer CNN is reduced to one convolution, and differences in model settings/protocol limit exact numerical comparability. CUDA execution, true cold start, new datasets, temporal generalization, review models and rating prediction remain untested; do not claim otherwise.

An automated PASS checks mechanics and evidence but cannot certify the truth of contribution statements, visual aesthetics or the instructor's grade.
