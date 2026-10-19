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
git pull --ff-only origin main
.venv\Scripts\python.exe scripts\final_preflight.py --roster-only
.venv\Scripts\python.exe scripts\finalize_submission.py --preview
```

Preview mode runs unit checks and experiment verification, rebuilds supporting documentation, report and editable presentation, executes the notebook, and checks exported document consistency. It preserves superseded ZIP/hash/manifest files privately under `.audit/submission-history/`. No final submission ZIP remains at the public deliverable path.

After inspecting the rebuilt PDF and all slides, record the current export hashes in the passing `.audit/visual_qa.json` inspection receipt and run `scripts/finalize_submission.py --verify-preview`. The technical audit supplies that receipt after actual inspection. This checks the current roster, roles, contribution placeholders, metric means and visual-review hashes without rebuilding.

The presenter study guide is at `docs/GROUP4_VIVA_PLAYBOOK.md`, with a compact revision sheet at `docs/VIVA_LAST_MINUTE.md`. Each student should rehearse their assigned slides and be ready for questions from the other sections too.

## Final build after authentic contributions are confirmed

Each member verifies the work they actually performed or reviewed. Add their accurate `contribution` statement to `configs/student.json` and set their `contribution_confirmed` field to `true` only after their explicit confirmation. Rebuild a preview with the updated attribution, inspect it, refresh the visual receipt and verify that preview. Then package those exact reviewed exports:

```powershell
.venv\Scripts\python.exe scripts\finalize_submission.py --package-only
```

The packaging step validates the four-person order, explicit contribution confirmation, matching attribution and metric means, and the visual receipt for the actual PDF/PPTX. It rebuilds `deliverables/submission.zip` without changing the reviewed exports or retraining. If the tool stops, investigate the exact blocker and do not claim submission readiness.

## Human approval and actual hand-in

- Confirm actual individual work attribution; do not assign credit based only on who is speaking.
- Examine PDF pages and rendered PowerPoint slides for readability, text overflow, plots, references and correct names/roll numbers.
- Confirm the department's submission format, deadline and any declarations, then personally submit through the required platform.

Deliverables: `deliverables/final_report.pdf`, `deliverables/CoupledCF_presentation.pptx`, `deliverables/submission.zip`, source scripts/notebook, requirements, README, tables and plots.

The ZIP intentionally excludes original MovieLens data, prepared arrays and selected `.pt` checkpoints because the original dataset has redistribution terms and checkpoints can contain derived feature buffers. The original local workspace keeps those for live demonstration; a reviewer can follow the README to rebuild from the official dataset.

The reported results are from five variants on one fixed MovieLens 1M split, three model seeds, ten CPU epochs, and 99 ranking negatives per user. The paper's original two-layer CNN is reduced to one convolution, and differences in model settings/protocol limit exact numerical comparability. CUDA execution, true cold start, new datasets, temporal generalization, review models and rating prediction remain untested; do not claim otherwise.

An automated PASS checks mechanics and evidence but cannot certify the truth of contribution statements, visual aesthetics or the instructor's grade.
