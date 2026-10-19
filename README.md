# CoupledCF on MovieLens 1M

This university mini project studies Zhang et al., *A Generic Framework for Learning Explicit and Implicit User-Item Couplings in Recommendation*, IEEE Access 7, 2019, DOI [10.1109/ACCESS.2019.2937841](https://doi.org/10.1109/ACCESS.2019.2937841). It implements the attribute-based Top-N recommendation component in PyTorch. It is a partial reproduction, with documented choices where the paper leaves implementation details unspecified.

## Set up

Use Python 3.10 or newer. Run commands from this directory. These PowerShell commands create an isolated environment. The download script obtains the original dataset from GroupLens and verifies its published checksum.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\prepare_data.py
.venv\Scripts\python.exe -m pytest -q
```

The experimental environment used Python 3.13 and CPU PyTorch. CUDA is supported when a CUDA-enabled PyTorch installation is available. Set `training.device` to `cuda` in a copied experiment configuration. An unavailable CUDA device raises an error instead of silently switching devices.

## Reproduce the experiments

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py --config configs\main.json
.venv\Scripts\python.exe scripts\summarize_results.py
.venv\Scripts\python.exe scripts\evaluate.py results\main\coupled_seed42
.venv\Scripts\python.exe scripts\execute_notebook.py
.venv\Scripts\python.exe scripts\verify_submission.py
```

Completed matching runs are preserved. To run a fresh experiment, copy `configs/main.json` and change `output_dir`. A conflicting existing run is an error. The pilot has its own configuration and directory and is excluded from the reported comparison.

The GitHub repository and submission ZIP contain measured run records but exclude checkpoints with MovieLens feature buffers. In a **fresh clone or extracted ZIP**, preserve those reference records under a different directory before running the main experiment:

```powershell
Move-Item -LiteralPath results\main -Destination results\reference_main
.venv\Scripts\python.exe scripts\run_experiments.py --config configs\main.json
```

This creates new complete runs at the paths used by the summarizer and verifier. The full local workspace already retains every checkpoint and can evaluate them directly. Existing metrics without their checkpoint are rejected rather than silently treated as reproducible completed runs. For a fresh study using a copied configuration, update the main configuration's output path before using the main-study summarizer and verifier.

The main configuration fixes the budget before the reported runs. It compares DeepCF, local CNN coupling, global coupling, combined local-plus-implicit coupling, and complete local-plus-global-plus-implicit CoupledCF. All variants share the same split and fixed ranking candidates. See `results/summary.csv` for measured means and seed variation, and `results/ablation.csv` for paired differences.

The recorded study used `scripts/run_study.py --jobs 2` to schedule at most two independent seed processes. Each process uses two PyTorch threads and writes separate run directories. `run_experiments.py` also supports `--seed 42` for a single configured seed. Run times reflect the observed scheduling and machine load and are not controlled speed comparisons.

`requirements-lock.txt` records the tested direct dependency versions. The CPU PyTorch wheel uses the official CPU package index. To recreate those pins, use `python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt`. PDF generation/source inspection additionally uses `requirements-artifacts.txt`. The editable PowerPoint builder uses Codex's bundled `@oai/artifact-tool` authoring runtime; the exported PPTX can be edited independently in PowerPoint.

## Data and evaluation

All 1,000,209 ratings count as observed interactions. Each user has one seeded random validation positive and one separate seeded random test positive. The remaining 988,129 positives are used for training. Four uniformly sampled unobserved negatives accompany each training positive. Evaluation ranks one target against 99 fixed unobserved items per user. Primary metrics are HR@10 and NDCG@10. Lower original item ID breaks exact score ties, independent of the target's candidate column.

The metadata contains 3,883 movies, including 3,706 rated movies. The maximum original movie ID is 3,952, which the paper lists as its item count. This implementation encodes the 3,883 actual metadata records contiguously and samples from that catalogue. That distinction limits numerical comparability with the paper.

The local split archive contains ID maps, feature encodings, the observed mask and all split/candidate arrays. Every completed local run preserves a split SHA256, configuration, source hashes, epoch history, selected checkpoint, checkpoint SHA256 and per-user ranks. The repository retains the run records, hashes and ranks. `data/processed/manifest.json` records raw file provenance and encoding vocabularies after dataset preparation.

## Deliverables

- `src/coupledcf/` contains data, models, ranking metrics and training logic.
- `scripts/` contains dataset, training, evaluation, results, notebook and verification entry points.
- `tests/` contains behavioral model, leakage and ranking checks.
- `notebooks/CoupledCF_reproduction.ipynb` is the runnable walkthrough. Its executed copy is retained separately.
- `docs/research/paper_analysis.md` analyzes the paper with PDF page references.
- `docs/methodology.md` explains the implementation and protocol adaptations.
- `docs/final_report.md` and `deliverables/final_report.pdf` contain the academic report.
- `deliverables/CoupledCF_presentation.pptx` contains the presentation.
- `docs/viva_questions.md` contains preparation questions and answers.
- `docs/requirements_checklist.md` maps assignment requirements to evidence.
- `results/` contains measured tables, plots, run histories and checkpoints.
- `decisions.tsv` records consequential choices and verification evidence.

## Scope and interpretation

This project evaluates warm users on one random split of MovieLens 1M. It does not establish true cold-start performance, temporal deployment quality, or generalization to other datasets. The review-text/Doc2Vec path, rating prediction and the paper's other datasets are analyzed rather than reproduced. One convolution replaces the paper's stated two CNN layers. The reduced epoch budget, feature projections, pooling choices and larger batch size are also documented adaptations. The experiments test whether the implemented explicit coupling improves this baseline under this budget. They do not certify an exact numerical reproduction of the paper.

The official Group 4 names and roll numbers are prefilled in `configs/student.json` from the supplied RS allocation. Each member must confirm their **actual individual contribution** there before submission. Regenerate and verify the exported report, slides and ZIP after confirming those statements; see `deliverables/submission_guide.md`. Do not use the currently committed placeholder-bearing exports as the personalized final hand-in.

## Data acknowledgment

MovieLens 1M comes from [GroupLens](https://grouplens.org/datasets/movielens/1m/). Cite F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015, DOI [10.1145/2827872](https://doi.org/10.1145/2827872). The original `data/raw/ml-1m/README` contains the license. Do not redistribute the dataset without the required permission. Git and the submission packaging script exclude raw and processed data, checkpoints containing MovieLens feature buffers, the virtual environment and private build files.
