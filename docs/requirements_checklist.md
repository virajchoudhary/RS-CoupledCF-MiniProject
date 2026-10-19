# Requirement to deliverable checklist

Coverage statuses describe verified project evidence, not marks awarded by the professor. The assignment rubric appears on page 4 of the supplied brief. The final verifier receipt is `.audit/verification.json`. Group 4 identities are populated from the official allocation sheet; actual individual contributions require human confirmation.

| Rubric | Component | Marks available | Evidence | Coverage |
| --- | --- | --- | --- | --- |
| 1 | Paper understanding and research problem | 1.5 | docs/research/paper_analysis.md; final_report Part A | PASS |
| 2 | Methodology and algorithm analysis | 1.5 | docs/methodology.md; final_report Part B; presentation architecture | PASS |
| 3 | Implementation of selected technique | 2.5 | src/coupledcf/models.py; tests/test_models.py; selected checkpoints | PASS |
| 4 | Experimental evaluation and comparison | 2.0 | 15 completed runs; results/summary.csv; ablation.csv; prediction replay | PASS |
| 5 | Critical analysis and proposed improvement | 1.5 | final_report critical analysis; hypotheses distinguished from measured findings | PASS |
| 6 | Report quality presentation and individual contribution | 1.0 | final_report PDF; editable PPTX; viva; configs/student.json | PARTIAL: actual individual contributions require student confirmation |

| ID | Requirement | Evidence | Coverage |
| --- | --- | --- | --- |
| A1 | Define and classify the recommendation problem | final_report Part A; paper_analysis problem and scope | PASS |
| A2 | Explain motivation and importance | final_report Part A motivation | PASS |
| A3 | Identify research gap | final_report Part A research gap | PASS |
| A4 | Identify claimed contributions | paper_analysis; final_report contributions | PASS |
| B1 | Identify all inputs | methodology input fields; final_report pipeline table | PASS |
| B2 | Explain feature extraction representation interaction and prediction | models.py; final_report Part B; presentation editable architecture | PASS |
| B3 | Explain output | raw logit and ranked items; sigmoid objective; final_report Part B | PASS |
| C1 | Implement a meaningful paper component directly | models.py outer product CNN and branch fusion | PASS |
| C2 | Mandatory baseline | DeepCF three matched seed runs; train-only popularity reference | PASS |
| C3 | Original dataset | official MovieLens 1M MD5; manifest; full counts and leakage checks | PASS |
| C4 | Ablation | five variants; 12 paired seed comparisons in ablation.csv | PASS |
| C5 | Strengths limitations and generalizability | final_report critical analysis | PASS |
| C6 | Compute training time GPU parameters and memory | results/runs.csv; per-run software and resource records; CPU/CUDA limitations | PASS |
| C7 | Data requirements and reproducibility | methodology; README; split/source/checkpoint hashes | PASS |
| C8 | Meaningful improvement | field-aware cross-attention proposal with controlled test design | PASS |
| S1 | Python code requirements and README | src/; scripts/; requirements.txt; README.md | PASS |
| S2 | Jupyter notebook or Python files | source and executed CoupledCF_reproduction notebooks | PASS |
| S3 | Real measured results | results/main/; summary.csv; runs.csv; figures/ | PASS |
| S4 | Presentation and viva | deliverables/CoupledCF_presentation.pptx; docs/viva_questions.md; docs/GROUP4_VIVA_PLAYBOOK.md; docs/VIVA_LAST_MINUTE.md | PASS |
| U1 | Read both PDFs including equations tables and diagrams | paper_analysis source page references; sources/assignment.txt; rendered source inspection | PASS |
| U2 | Explicit implicit local/global DeepCF and full architecture | paper_analysis; methodology; implemented variants | PASS |
| U3 | Deterministic encoding sampling split and leakage checks | data.py; manifest.json; tests/test_data.py | PASS |
| U4 | CPU and CUDA pipeline Adam BCE validation and checkpointing | engine.py; configs/main.json; tests/test_engine.py; CPU executed and CUDA code-only | PASS |
| U5 | Timing resources parameter counts and exceptions | per-run result.json; engine.py finite-loss and device guards | PASS |
| U6 | HR NDCG at ten and secondary K | metrics.py; K=5,10,20 in every saved result | PASS |
| U7 | Fair candidates and controls across models | one shared split hash; deterministic per-epoch sampling; verifier config equality | PASS |
| U8 | Discuss published comparison and deviations | published Table7 in references.json and report; all architecture/protocol adaptations disclosed | PASS |
| U9 | Transparent negative findings and hypotheses | report interprets actual delta signs; ablation and curve analysis; no statistical-significance claim | PASS |
| U10 | Saved configurations metrics figures and checkpoints | all 15 completed run bundles in results/main/ | PASS |
| U11 | Execute tests smoke actual training evaluation and notebook | tests.log; engine_tests.log; pilot; main runs; notebook_execution.json; verification.json | PASS |
| U12 | Final artifacts match actual results | generated report/slides; summary reaggregation and checkpoint replay; visual QA records | PASS |

CUDA execution is unverified, so U4 covers a working CPU pipeline and the CUDA code path, not demonstrated GPU results. The review/Doc2Vec path, rating prediction and other datasets are analyzed rather than reproduced. A single convolution replaces the paper's stated two CNN layers. The assignment requests a partial reproduction, and these boundaries are visible in the report and presentation.

Group 4 names and roll numbers are now populated; actual contributions and approval of regenerated final exports remain pending human confirmation. The official RS.xlsx lists Dhruva Ambhaikar (I002), Viraj Choudhary (I012), Rusheel Sharma (I069), and Shawn Mendes (I077).
