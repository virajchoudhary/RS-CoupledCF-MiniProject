# Submission guide

Before hand-in, fill the verified student names, roll numbers and actual individual contributions in `configs/student.json`. The report, presentation and viva must describe the work each student can explain. No student-specific information has been invented.

Changing that configuration does not update existing exports automatically. Regenerate the report with `scripts/build_report.py` after installing `requirements-artifacts.txt`, and update the editable presentation placeholders. Refresh the document/visual checks and submission archive after changing exported files.

The academic deliverables are `final_report.pdf` and `CoupledCF_presentation.pptx`. Editable report text, paper analysis, methodology, viva answers, the complete requirement checklist, the runnable/executed notebook and measured experiment evidence are in the corresponding project directories.

The complete local project retains all selected checkpoints in `results/main/*/best.pt`. Each run has its configuration, history, source snapshot, result, hashes and per-user ranks. The original official dataset and prepared split remain local for direct checkpoint replay.

`submission.zip` is produced only after the final evidence checks pass. It is the source and results package. It excludes raw/processed MovieLens data and checkpoints containing dataset feature buffers, along with the virtual environment, private caches, pilot and interrupted runs. GroupLens' original README states that dataset redistribution requires separate permission. A recipient can download the official archive and regenerate the split and training with the documented commands. The split SHA256 in `dataset_provenance.json` identifies the tested split. Keep the full local workspace for a live demonstration or direct checkpoint evaluation.

The experiment is a partial reproduction. It uses one convolution in place of the paper's stated two, disclosed feature projections, a separate validation target per user, a ten-epoch CPU budget and a larger batch. Every observed rating counts as an interaction. All five variants share the split and candidate arrays. CUDA execution, true cold start, temporal testing, the review path and rating prediction are not empirically certified by this project.

The final verification receipt accompanies this package. It checks the finished runs and checkpoint predictions, artifact presence and executed notebook. It does not establish an individual's academic contribution or an awarded rubric score.

To retrain from an extracted archive, first preserve its checkpoint-free reference records with `Move-Item -LiteralPath results\main -Destination results\reference_main`. Then follow the README setup and main experiment commands. New checkpoints and results will be written to `results/main`, which is the path read by the main-study summarizer and verifier. This step is unnecessary for direct replay in the original full local workspace.
