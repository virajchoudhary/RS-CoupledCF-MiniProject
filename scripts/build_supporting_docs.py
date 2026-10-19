"""Write viva answers and final requirement mapping from measured artifacts."""
from pathlib import Path
import csv
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/main.json').read_text())
    evidence_path = ROOT / '.audit/experiment_verification.json'
    if not evidence_path.exists():
        raise RuntimeError('Run verify_submission.py --experiments-only before marking the checklist complete')
    verified = json.loads(evidence_path.read_text())
    expected_runs = len(config['variants']) * len(config['seeds'])
    if verified.get('status') != 'PASS' or len(verified.get('runs', [])) != expected_runs:
        raise RuntimeError('Final checklist requires a passing complete experiment-verification receipt')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if verified['study_config_sha256'] != digest(ROOT / 'configs/main.json'):
        raise RuntimeError('Experiment verification does not match the current study configuration')
    for record in verified['runs']:
        directory = ROOT / config['output_dir'] / record['name']
        if record['result_sha256'] != digest(directory / 'result.json') or record['checkpoint_sha256'] != digest(directory / 'best.pt'):
            raise RuntimeError(f'Run artifacts changed after verification: {record["name"]}')
    for name in ['notebook_execution.json', 'document_consistency.json', 'visual_qa.json']:
        path = ROOT / '.audit' / name
        if not path.exists():
            raise RuntimeError(f'Final checklist requires {name}')
        receipt = json.loads(path.read_text())
        if receipt.get('status') != 'PASS':
            raise RuntimeError(f'Final checklist requires a passing {name}')
    visual = json.loads((ROOT / '.audit/visual_qa.json').read_text())
    for name, key in [('final_report.pdf', 'pdf_sha256'), ('CoupledCF_presentation.pptx', 'pptx_sha256')]:
        if digest(ROOT / 'deliverables' / name) != visual[key]:
            raise RuntimeError(f'Visual review is stale for {name}')
    with (ROOT / 'results/summary.csv').open() as stream:
        rows = {row['variant']: row for row in csv.DictReader(stream)}
    delta = float(rows['coupled']['NDCG@10_mean']) - float(rows['deepcf']['NDCG@10_mean'])
    outcome = 'higher' if delta > 0 else 'lower'
    qa = [
        ('What problem does the implemented model solve?', 'Personalized Top-N ranking from implicit user-item interactions. It scores candidate items for known users. The original paper also evaluates rating regression, which this project analyzes but does not reproduce.'),
        ('What research gap motivates CoupledCF?', 'The paper aims to learn explicit cross-feature interactions and implicit latent ID interactions jointly. Its CNN learns local patterns in an outer-product matrix, while a separate flattened path retains global interactions.'),
        ('What is explicit coupling?', 'It is the learned interaction between observed side attributes after embedding. In this project user demographics interact with movie genre representations. Explicit coupling does not mean explicit star-rating supervision.'),
        ('What is implicit coupling?', 'It is learned from user and item ID factors and interaction history. DeepCF multiplies the two ID embeddings elementwise and transforms the result through a neural network.'),
        ('How does the outer product differ from an elementwise product?', 'Two length-eight vectors produce 64 cross-coordinate products in an 8 by 8 outer product. An elementwise product produces only eight matching-coordinate products. The explicit branch uses the former, and the implicit branch uses the latter with width 32.'),
        ('What is local versus global coupling?', 'Local coupling applies convolution and max pooling to the coupling matrix. Global coupling flattens the original matrix. The complete model combines both with the implicit vector.'),
        ('What shapes pass through the implemented local branch?', 'The matrix is [batch,1,8,8]. A valid 3 by 3 convolution with eight channels gives [batch,8,6,6]. Max pooling with size two gives [batch,8,3,3], then flattening gives 72 values.'),
        ('Is this the exact paper architecture?', 'No. It reproduces the selected outer-product CNN mechanism and branch structure. The paper states two CNN layers; this bounded partial reproduction uses one. Feature projections, pooling geometry, validation allocation, batch size, framework and epoch budget are also disclosed differences.'),
        ('Why use MovieLens 1M?', 'It is one of the original paper datasets, publicly available from GroupLens, and supplies both interaction history and the demographic/genre side information required for the selected technique.'),
        ('How many observations and entities are used?', 'The source contains 1,000,209 ratings and 6,040 users. Metadata contains 3,883 movies, of which 3,706 have ratings. The maximum movie ID is 3,952, which differs from the count of actual metadata rows.'),
        ('Are one-star and two-star ratings negatives?', 'No. In the chosen paper-style implicit protocol every recorded rating is a positive observed interaction. Negatives are sampled from unobserved user-item pairs. This predicts interaction, not rating intensity.'),
        ('How are multiple movie genres encoded?', 'All 18 genre flags are retained in a multi-hot vector, which is projected to the side-vector width. The code does not select only the first listed genre.'),
        ('How are splits and negatives protected from leakage?', 'Two distinct random observed items per user are held out for validation and test. Training contains all other positives. The complete observed mask excludes every known positive from negative pools. Candidate lists are unique and fixed. Automated checks deliberately insert leakage and require rejection.'),
        ('Does the protocol simulate future recommendations?', 'No. It is random holdout and may train on later events than the test event. A chronological split would test a different deployment question and must be reported separately.'),
        ('Why use four training negatives?', 'The paper uses four uniformly sampled unobserved negatives per positive. This project matches that ratio and resamples each epoch. Sampling with replacement permits repeated negatives.'),
        ('Why evaluate against 99 negatives?', 'The paper ranks one held-out target and 99 sampled unobserved items per user. The total is 100 candidates. Using only nine negatives would make HR@10 trivial and would not match the paper.'),
        ('What are HR@10 and NDCG@10?', 'HR@10 is the fraction of target items ranked in the first ten positions. NDCG@10 rewards those hits by 1/log2(rank+1), so a hit near rank one counts more. With one positive, ideal DCG equals one.'),
        ('What happens to exact score ties?', 'Items with lower original ID rank first. This stable rule is independent of the positive column and is shared across all models.'),
        ('Why train with BCEWithLogitsLoss?', 'It is a numerically stable form of sigmoid plus binary cross entropy, matching the interaction-classification objective. Logit and sigmoid rankings are identical because sigmoid is strictly increasing.'),
        ('How is the best checkpoint chosen?', 'The highest validation NDCG@10 selects the epoch, retaining the earliest epoch on a tie. Test scores do not choose the checkpoint. The saved state is reloaded before final evaluation.'),
        ('What makes the comparisons fair?', 'All variants share the full dataset, split, candidates, seeds, optimizer settings, epoch budget and deterministic per-epoch negative sampling. Parameter counts differ because branches are removed, so capacity is a remaining confounder.'),
        ('What do the actual results show?', f'CoupledCF has mean HR@10 {float(rows["coupled"]["HR@10_mean"]):.4f} and NDCG@10 {float(rows["coupled"]["NDCG@10_mean"]):.4f}. Its NDCG@10 is {abs(delta):.4f} {outcome} than DeepCF. Read results/summary.csv for all five variants and sample standard deviations, and results/ablation.csv for paired differences.'),
        ('Why might the combined model underperform?', 'Possible causes include a shorter optimization budget, larger batches, coarse genre information, arbitrary CNN coordinate locality, dropout and fusion optimization. The measured ablations and curves describe behavior, but they do not prove which of these hypotheses caused a gap.'),
        ('Does a larger score than the paper prove a better implementation?', 'No. Catalogue, split, candidate draws, training updates and architecture differ. Published Table 7 is a reference under a different experimental setup, not an apples-to-apples ranking of implementations.'),
        ('What does three-seed standard deviation mean?', 'It describes model-seed variation on one fixed split. It is not uncertainty over datasets or split choices, and it does not by itself establish statistical significance.'),
        ('Is cold start solved?', 'No cold-start experiment was run. All evaluated users have training histories. Side features may help new entities in principle, but the combined model has fixed ID tables and no validated unknown-ID policy.'),
        ('How does the review-based path in the paper differ?', 'It aggregates reviews into user and item documents and uses Doc2Vec PV-DM vectors as side information. This project uses metadata and does not claim to have trained Doc2Vec.'),
        ('What compute is needed?', 'The completed study uses CPU PyTorch with two threads. CUDA is supported in the pipeline when available but was not executed. Actual parameter counts, run times and sampled process RSS are saved for every run. These timings are observations, not controlled hardware speedup comparisons.'),
        ('What improvement is proposed?', 'Field-aware cross attention could model named demographic fields and genre tokens without assuming local neighborhoods in arbitrary embedding coordinates. A parameter-matched evaluation with the same split/seeds and additional cold-start protocols would test the hypothesis. It is not a measured improvement.'),
        ('How can another person reproduce a reported number?', 'Download and verify the official archive, prepare the split, create the documented environment, then run the fixed config or evaluate a retained checkpoint. Hashes, source snapshots, configuration, histories and per-user ranks identify every completed result.'),
        ('What did each student contribute?', 'Group 4 names and roll numbers are prefilled from the official RS allocation. Each student must confirm their actual contribution in configs/student.json and be able to explain that work. Never invent academic contributions.'),
    ]
    text = '# Viva questions and answers\n\n'
    for index, (question, answer) in enumerate(qa, 1):
        text += f'## {index}. {question}\n\n{answer}\n\n'
    (ROOT / 'docs/viva_questions.md').write_text(text, encoding='utf-8')
    rubric = [
        ('1', 'Paper understanding and research problem', '1.5', 'docs/research/paper_analysis.md; final_report Part A', 'PASS'),
        ('2', 'Methodology and algorithm analysis', '1.5', 'docs/methodology.md; final_report Part B; presentation architecture', 'PASS'),
        ('3', 'Implementation of selected technique', '2.5', 'src/coupledcf/models.py; tests/test_models.py; selected checkpoints', 'PASS'),
        ('4', 'Experimental evaluation and comparison', '2.0', f'{expected_runs} completed runs; results/summary.csv; ablation.csv; prediction replay', 'PASS'),
        ('5', 'Critical analysis and proposed improvement', '1.5', 'final_report critical analysis; hypotheses distinguished from measured findings', 'PASS'),
        ('6', 'Report quality presentation and individual contribution', '1.0', 'final_report PDF; editable PPTX; viva; configs/student.json', 'PARTIAL: actual individual contributions require student confirmation'),
    ]
    text = '# Requirement to deliverable checklist\n\n'
    text += ('Coverage statuses describe verified project evidence, not marks awarded by the professor. '
             'The assignment rubric appears on page 4 of the supplied brief. The final verifier receipt is '
             '`.audit/verification.json`. Group 4 identities are populated from the official allocation sheet; actual individual contributions require human confirmation.\n\n')
    text += '| Rubric | Component | Marks available | Evidence | Coverage |\n| --- | --- | --- | --- | --- |\n'
    text += '\n'.join('| ' + ' | '.join(row) + ' |' for row in rubric) + '\n\n'
    items = [
        ('A1', 'Define and classify the recommendation problem', 'final_report Part A; paper_analysis problem and scope'),
        ('A2', 'Explain motivation and importance', 'final_report Part A motivation'),
        ('A3', 'Identify research gap', 'final_report Part A research gap'),
        ('A4', 'Identify claimed contributions', 'paper_analysis; final_report contributions'),
        ('B1', 'Identify all inputs', 'methodology input fields; final_report pipeline table'),
        ('B2', 'Explain feature extraction representation interaction and prediction', 'models.py; final_report Part B; presentation editable architecture'),
        ('B3', 'Explain output', 'raw logit and ranked items; sigmoid objective; final_report Part B'),
        ('C1', 'Implement a meaningful paper component directly', 'models.py outer product CNN and branch fusion'),
        ('C2', 'Mandatory baseline', 'DeepCF three matched seed runs; train-only popularity reference'),
        ('C3', 'Original dataset', 'official MovieLens 1M MD5; manifest; full counts and leakage checks'),
        ('C4', 'Ablation', 'five variants; 12 paired seed comparisons in ablation.csv'),
        ('C5', 'Strengths limitations and generalizability', 'final_report critical analysis'),
        ('C6', 'Compute training time GPU parameters and memory', 'results/runs.csv; per-run software and resource records; CPU/CUDA limitations'),
        ('C7', 'Data requirements and reproducibility', 'methodology; README; split/source/checkpoint hashes'),
        ('C8', 'Meaningful improvement', 'field-aware cross-attention proposal with controlled test design'),
        ('S1', 'Python code requirements and README', 'src/; scripts/; requirements.txt; README.md'),
        ('S2', 'Jupyter notebook or Python files', 'source and executed CoupledCF_reproduction notebooks'),
        ('S3', 'Real measured results', 'results/main/; summary.csv; runs.csv; figures/'),
        ('S4', 'Presentation and viva', 'deliverables/CoupledCF_presentation.pptx; docs/viva_questions.md; docs/GROUP4_VIVA_PLAYBOOK.md'),
        ('U1', 'Read both PDFs including equations tables and diagrams', 'paper_analysis source page references; sources/assignment.txt; rendered source inspection'),
        ('U2', 'Explicit implicit local/global DeepCF and full architecture', 'paper_analysis; methodology; implemented variants'),
        ('U3', 'Deterministic encoding sampling split and leakage checks', 'data.py; manifest.json; tests/test_data.py'),
        ('U4', 'CPU and CUDA pipeline Adam BCE validation and checkpointing', 'engine.py; configs/main.json; tests/test_engine.py; CPU executed and CUDA code-only'),
        ('U5', 'Timing resources parameter counts and exceptions', 'per-run result.json; engine.py finite-loss and device guards'),
        ('U6', 'HR NDCG at ten and secondary K', 'metrics.py; K=5,10,20 in every saved result'),
        ('U7', 'Fair candidates and controls across models', 'one shared split hash; deterministic per-epoch sampling; verifier config equality'),
        ('U8', 'Discuss published comparison and deviations', 'published Table7 in references.json and report; all architecture/protocol adaptations disclosed'),
        ('U9', 'Transparent negative findings and hypotheses', 'report interprets actual delta signs; ablation and curve analysis; no statistical-significance claim'),
        ('U10', 'Saved configurations metrics figures and checkpoints', 'all 15 completed run bundles in results/main/'),
        ('U11', 'Execute tests smoke actual training evaluation and notebook', 'tests.log; engine_tests.log; pilot; main runs; notebook_execution.json; verification.json'),
        ('U12', 'Final artifacts match actual results', 'generated report/slides; summary reaggregation and checkpoint replay; visual QA records'),
    ]
    text += '| ID | Requirement | Evidence | Coverage |\n| --- | --- | --- | --- |\n'
    text += '\n'.join(f'| {i} | {r} | {e} | PASS |' for i, r, e in items)
    text += ('\n\nCUDA execution is unverified, so U4 covers a working CPU pipeline and the CUDA code path, '
             'not demonstrated GPU results. The review/Doc2Vec path, rating prediction and other datasets '
             'are analyzed rather than reproduced. A single convolution replaces the paper\'s stated two '
             'CNN layers. The assignment requests a partial reproduction, and these boundaries are visible '
             'in the report and presentation.\n\nGroup 4 names and roll numbers are now populated; actual contributions '
             'and regenerated final exports remain pending human confirmation. The official RS.xlsx lists '
             'Dhruva Ambhaikar (I002), Viraj Choudhary (I012), Rusheel Sharma (I069), and Shawn Mendes (I077).\n')
    (ROOT / 'docs/requirements_checklist.md').write_text(text, encoding='utf-8')
    print('Generated viva and final requirement mapping')


if __name__ == '__main__':
    main()
