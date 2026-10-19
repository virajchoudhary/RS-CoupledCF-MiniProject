# Group 4: final submission handoff

This is a technical-completion handoff for the **Research Paper Study and Partial Reproduction** of CoupledCF. It is not an assertion that the instructor has approved or graded the project.

## Already verified from the official RS allocation

| Student | Roll number |
| --- | --- |
| Dhruva Ambhaikar | I002 |
| Viraj Choudhary | I012 |
| Rusheel Sharma | I069 |
| Shawn Mendes | I077 |

Course: Recommendation Systems. Group: 4. Institution: SVKM's NMIMS MPSTME, Mumbai.

## Assigned viva responsibilities (not proof of work completed)

| Roll | Student | Slides | Preparation ownership |
| --- | --- | --- | --- |
| I002 | Dhruva Ambhaikar | 1–3 | Motivation, problem and gap |
| I012 | Viraj Choudhary | 4–7 | Architecture, equations, implementation |
| I069 | Rusheel Sharma | 8–12 | Data, training, metrics and experiments |
| I077 | Shawn Mendes | 13–16 | Published comparison, ablations, limitations and conclusion |

For speaker notes, likely questions, mathematical explanations, and a complete rehearsal runbook, read [GROUP4_VIVA_PLAYBOOK.md](GROUP4_VIVA_PLAYBOOK.md) and [VIVA_LAST_MINUTE.md](VIVA_LAST_MINUTE.md).

## Only human-dependent tasks

1. **Each student confirms their actual individual contribution.** Replace their bracketed `contribution` field in `configs/student.json` and set `contribution_confirmed` to `true` only after their explicit confirmation. Describe work each person truly performed or reviewed and can defend in the viva. Do not infer work solely from authorship, speaking assignments, agent logs, or the existence of a file.
2. **The group reviews the final report and presentation.** Check that the attribution is fair and all four members can explain their assigned portions. Review the final slides visually for table wrapping, chart labels, citations and readability. Correct any discovered mistakes.
3. **The group performs the actual institutional hand-in** in the required platform, with any student declarations or professor-specific approval.

## Everything else is automated in the original project workspace

The original local workspace is required: it holds the officially downloaded MovieLens dataset and 15 selected checkpoints that were deliberately excluded from GitHub/submission ZIP due to licensing and feature-buffer content.

1. Pull the finalized repository changes into that workspace (preserve local `data/` and `results/main/*/best.pt`).
2. Confirm individual contributions in `configs/student.json`.
3. With the project's Python environment and Codex presentation runtime active, first regenerate a **non-submittable preview** while contributions remain unconfirmed, then perform the final build only after genuine contributions are recorded:

```powershell
.venv\Scripts\python.exe scripts\final_preflight.py --roster-only
.venv\Scripts\python.exe scripts\finalize_submission.py --preview
# After inspecting the rebuilt PDF and every slide, refresh the visual receipt:
.venv\Scripts\python.exe scripts\finalize_submission.py --verify-preview
# Only after truthful contributions are confirmed and the updated exports are reviewed:
.venv\Scripts\python.exe scripts\finalize_submission.py --package-only
```

Preview mode rebuilds research support docs, report, notebook and presentation and checks experimental evidence and tests. It archives superseded ZIP/hash/manifest files privately so an older package cannot be mistaken for the current submission. The preview verification requires a visual receipt tied to the actual export hashes. Package-only mode checks explicit contribution confirmation, matching attribution, slide roles, metrics and the current visual receipt before creating `deliverables/submission.zip`. Neither mode retrains or replaces the research results.

Do not manually bypass the final preflight or copy the previous placeholder-bearing PDF, deck, or ZIP into a hand-in. If the automated finalization stops, report the exact failure and fix its cause rather than claiming PASS.

## Evidence and scope

The completed study contains five model variants by three seeds, 15 total runs. Mean full CoupledCF scores are HR@10 0.7937 and NDCG@10 0.5224. DeepCF baseline scores are 0.7825 and 0.5022. See the current technical audit and `deliverables/preview_status.json` for the tested source and inspected export status.

This is a **partial reproduction**. The 10-epoch CPU budget, one CNN layer, side-feature projections and evaluation protocol adaptations remain explicit limitations. The project does not claim an exact paper reproduction, a controlled GPU result, a cold-start result, or a statistically significant superiority claim.

Group approval of attribution and the final physical/online submission cannot be performed by software.
