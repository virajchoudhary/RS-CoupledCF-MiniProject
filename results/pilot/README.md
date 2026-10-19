# Preliminary full-data pilot

The pilot completed one epoch before the final paper-fidelity correction to explicit-then-implicit fusion. It established that the full dataset, negative sampling, optimizer, checkpoint and ranking pipeline executed on CPU. Its architecture used preliminary flat fusion, so it is excluded from every final comparison, ablation, report table and submission archive.

The final combined models include an explicit MLP before joining DeepCF. Use results/main checkpoints with the current evaluation script. Pilot and interrupted checkpoints are retained as historical evidence and should not be loaded as current-architecture models. The pilot's original configuration, hashes, selected state, history and ranks remain unchanged.
