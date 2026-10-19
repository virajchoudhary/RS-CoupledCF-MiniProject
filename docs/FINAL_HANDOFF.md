# Group 4: final submission handoff

This is a technical-completion handoff for the **Research Paper Study and Partial Reproduction** of CoupledCF. It is not an assertion that the instructor has approved or graded the project.

## Already verified from the official RS allocation

| Student | Roll number |
| --- | --- |
| Dhruva Ambhaikar | I002 |
| Rusheel Sharma | I069 |
| Viraj Choudhary | I012 |
| Shawn Mendes | I077 |

Course: Recommendation Systems. Group: 4. Institution: SVKM's NMIMS MPSTME, Mumbai.

## Only human-dependent tasks

1. **Each student confirms their actual individual contribution.** Replace the four bracketed `contribution` fields in `configs/student.json`. Describe work each person truly performed or reviewed and can defend in the viva. Do not infer work solely from authorship, agent logs, or the existence of a file. Do not invent names or academic credit.
2. **The group reviews the final report and presentation.** Check that the attribution is fair and all four members can explain their assigned portions. Review the final slides visually for table wrapping, chart labels, citations and readability. Correct any discovered mistakes.
3. **The group performs the actual institutional hand-in** in the required platform, with any student declarations or professor-specific approval.

## Everything else is automated in the original project workspace

The original local workspace is required: it holds the officially downloaded MovieLens dataset and 15 selected checkpoints that were deliberately excluded from GitHub/submission ZIP due to licensing and feature-buffer content.

1. Pull the finalized repository changes into that workspace (preserve local `data/` and `results/main/*/best.pt`).
2. Confirm individual contributions in `configs/student.json`.
3. With the project's Python environment and Codex presentation runtime active, execute:

```powershell
.venv\Scripts\python.exe scripts\final_preflight.py --roster-only
.venv\Scripts\python.exe scripts\finalize_submission.py
```

The second command runs Python tests, re-verifies actual experiments, rebuilds the research support docs, report, notebook and PowerPoint, executes notebook checks, cross-checks PDF/PPTX metrics, reruns the submission verifier, validates student attribution in exports, and rebuilds `deliverables/submission.zip`. It **does not retrain** or replace the reported experiment results.

Do not manually bypass the final preflight or copy the previous placeholder-bearing PDF, deck, or ZIP into a hand-in. If the automated finalization stops, report the exact failure and fix its cause rather than claiming PASS.

## Evidence and scope

The recorded completed study contains five model variants by three seeds, 15 total runs. The mean full CoupledCF scores are HR@10 0.7937 and NDCG@10 0.5224. The DeepCF baseline scores are 0.7825 and 0.5022. Recorded checks predate these finalization edits; the newly introduced hand-in tests and rebuilding must run on the original local environment before declaring final hand-in readiness.

This is a **partial reproduction**. The 10-epoch CPU budget, one CNN layer, side-feature projections and evaluation protocol adaptations remain explicit limitations. The project does not claim an exact paper reproduction, a controlled GPU result, a cold-start result, or a statistically significant superiority claim.

Group approval of attribution and the final physical/online submission cannot be performed by software.
