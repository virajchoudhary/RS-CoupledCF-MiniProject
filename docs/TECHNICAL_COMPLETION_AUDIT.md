# Technical completion and submission readiness audit

Audit date: 19 October 2026. Scope: the assigned CoupledCF paper and the
professor's four-page, 10-mark assignment brief.

**Technical preview: VERIFIED. Presentation material: READY for group rehearsal.
Institutional submission: BLOCKED by four unconfirmed contribution statements
and human approval.** These statuses describe evidence, not marks awarded.

## Preservation and synchronization

The clean local checkout was fast-forwarded from `445b70b` to merged main
`00005ac74d7778fab78d8f6c3c5c4a224c08fd2b`. No reset or research-result overwrite
was used. Hash checks confirm all 203 existing files under `data/` and
`results/` are unchanged, including raw data, the prepared split, checkpoints,
main results, pilot and interrupted-run evidence.

The previous report, deck, documents and notebook were backed up privately
under `.audit/final-completion/`. Superseded submission ZIP, checksum and
artifact manifest files were preserved under `.audit/submission-history/`
and removed from the current public deliverable paths. The old ZIP used a
two-person placeholder roster and was not a valid Group 4 hand-in.

## Assignment rubric

| Requirement | Marks available | Status | Current evidence |
| --- | ---: | --- | --- |
| Paper understanding and research problem | 1.5 | PASS | Paper analysis; report Part A; slides 1-4 |
| Methodology and algorithm analysis | 1.5 | PASS | Methodology; report Part B; equations and editable architecture on slides 4-7 |
| Implementation of selected technique | 2.5 | PASS | Direct PyTorch outer-product CNN and five branch variants; forward/backward and checkpoint tests |
| Experimental evaluation and comparison | 2.0 | PASS | Full MovieLens 1M; 15 completed runs; DeepCF and popularity references; metrics and paired ablations |
| Critical analysis and proposed improvement | 1.5 | PASS | Report; slides 13-15; limitations and untested field-aware attention proposal |
| Report quality, presentation and individual contribution | 1.0 | PARTIAL | Verified ten-page PDF and editable 16-slide deck; viva material complete; contributions pending |

Parts A, B and C and the complete detailed requirement mapping are in
`docs/requirements_checklist.md`.

## Scientific evidence

Official MovieLens archive MD5 matches
`c4d9eecfca2ab87c1945afe126590906`. All four extracted raw-file SHA256 values
match the manifest. The full set has 1,000,209 interactions, 6,040 users and
3,883 metadata movies, including 3,706 rated movies.

Split seed 2026 gives 988,129 training positives and 6,040 validation and
6,040 separate test positives. Every evaluation row has one target and
99 unique unobserved candidates. Split reconstruction and exclusion checks
pass. All models share the split and candidates.

All five models completed ten CPU epochs for seeds 42, 43 and 44.
The earliest maximum validation NDCG@10 selects the checkpoint. Fresh
prediction replay from all 15 checkpoints reproduced the exact saved test
ranks and HR/NDCG values. Summary means, sample standard deviations, parameter
counts, recorded costs and all 12 paired ablation rows recompute correctly.
No model was retrained during this audit.

| Model | Mean HR@10 | Mean NDCG@10 |
| --- | ---: | ---: |
| DeepCF | 0.7825 | 0.5022 |
| Local explicit | 0.3841 | 0.2142 |
| Global explicit | 0.3924 | 0.2151 |
| Local plus implicit | 0.7927 | 0.5203 |
| Full CoupledCF | 0.7937 | 0.5224 |

Full CoupledCF improves on DeepCF in all three matched seeds. The incremental
global branch over local plus implicit has mixed signs across seeds.
Three model seeds on one fixed split do not establish statistical significance.

The three earliest branch-only seed-42 runs retain earlier model/engine
snapshots. Their saved hashes match those copies. Later changes concern
combined-branch fusion, output-creation guards, validation and a CUDA setting.
Current CPU evaluation reproduces all saved ranks. See `docs/methodology.md`;
no claim of byte-identical training source across every run is made.

## Executed technical checks

| Check | Outcome |
| --- | --- |
| Full unit and regression suite | PASS, 53 cases |
| Python compilation and presentation builder syntax | PASS |
| Official archive, raw-file hashes and protected research files | PASS |
| Full split reconstruction, candidate uniqueness and leakage checks | PASS |
| Fifteen fresh checkpoint prediction replays | PASS, exact saved test ranks |
| Main configuration, run status, histories, selected epochs and source snapshots | PASS |
| Ranking/cost aggregates and paired ablations | PASS |
| Notebook execution | PASS, all six code cells, one bounded disposable optimizer update and retained checkpoint replay |
| Report/deck document consistency | PASS, all five displayed HR@10/NDCG@10 means match measurements |
| Four-person order and slide-topic/range checks | PASS |
| Viva playbook and revision sheet against results and deck | PASS |
| Final PowerPoint package, native charts and geometry | PASS, three editable charts; zero findings or warnings |
| Visual inspection | PASS, all ten report pages and all 16 exported slides |
| Current preview artifact fingerprints | PASS |
| Final workflow with pending contributions | Correctly STOPPED |
| Direct ZIP packager with pending contributions | Correctly REJECTED |
| ZIP staging failure and synthetic archive integrity checks | PASS; no student confirmation or final submission gate bypassed |
| Recorded scientific dependency versions | PASS, all ten unchanged |
| Broad environment `pip check` | WARNING, unrelated inherited global-package conflicts |

The existing `.venv` inherits system packages. Broad `pip check` reports
missing `requests` for unrelated installed packages and a Streamlit
`packaging` constraint. The project checks and complete preview run pass.
The audit did not change unrelated environments. PDF helpers installed
locally are ReportLab 4.5.1 and pypdf 6.21.1.

## Inspection and corrected defects

The PDF and deck were inspected as rendered exports, separately from their
generators. Equations, tensor sizes, tables, references, student details,
charts and scientific limits match the implementation and measured results.
The report's orphan last page was removed. Slide 10's footnote was repositioned
to clear a native-table geometry warning. Fifteen other slide renders and the
first nine final report-page renders matched already inspected versions.

PowerPoint itself was not opened. Package checks, embedded editable chart
data and reimport rendering passed. The group should test the deck on the
presentation-room computer during rehearsal.

The workflow now stages and validates presentation and ZIP outputs before
replacement, archives stale packages, requires explicit contribution
confirmation, checks actual slide topics, and binds visual approval to the
current PDF and deck hashes. Regression evidence includes failing-before
tests for absent safeguards and the contribution-text-only acceptance defect.

## Current artifacts

| Path | Exact use/status |
| --- | --- |
| `deliverables/final_report.pdf` | Ten-page verified PREVIEW; actual contributions explicitly pending |
| `deliverables/CoupledCF_presentation.pptx` | Editable 16-slide verified presentation preview |
| `docs/final_report.md` | Editable report generated with the PDF |
| `notebooks/CoupledCF_reproduction.ipynb` | Runnable source notebook with ordered roster |
| `notebooks/CoupledCF_reproduction.executed.ipynb` | Six successfully executed code cells |
| `docs/GROUP4_VIVA_PLAYBOOK.md` | Full speaker preparation and viva guide |
| `docs/VIVA_LAST_MINUTE.md` | Compact revision sheet |
| `docs/viva_questions.md` | Generated technical questions and answers |
| `deliverables/verification.json` | Passing 15-run and deliverable evidence |
| `deliverables/preview_status.json` | Current hashed preview inventory, `submission_ready=false` |
| `deliverables/submission.zip` | Deliberately ABSENT until final human gates clear |
| `.audit/final-completion/` | Private logs, backups, routing and preservation evidence |

## Human-only completion

| Student | Roll | Assigned slides | Actual contribution confirmation |
| --- | --- | --- | --- |
| Dhruva Ambhaikar | I002 | 1-3 | PENDING |
| Viraj Choudhary | I012 | 4-7 | PENDING |
| Rusheel Sharma | I069 | 8-12 | PENDING |
| Shawn Mendes | I077 | 13-16 | PENDING |

Each student must supply a truthful contribution statement and explicitly
confirm it. Only then may their `contribution_confirmed` field become `true`.
Speaking assignments do not establish completed work. The group must approve
the attribution, rehearse, test the presentation machine, and satisfy the
institution's deadline, format and declaration requirements.

After changing attribution, rebuild and review the exports again. A fresh
visual receipt must identify the inspected PDF/PPTX hashes:

```powershell
.venv\Scripts\python.exe scripts\finalize_submission.py --preview
.venv\Scripts\python.exe scripts\finalize_submission.py --verify-preview
.venv\Scripts\python.exe scripts\finalize_submission.py --package-only
```

The last command is gated and packages the reviewed exports without
rebuilding them. Inspect the produced ZIP and sidecar before institutional
submission. CUDA execution, true cold start, temporal evaluation, additional
datasets, rating regression and review/Doc2Vec experiments remain outside
this completed partial reproduction.
