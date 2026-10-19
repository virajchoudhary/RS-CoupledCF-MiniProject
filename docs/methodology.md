# Implementation and experimental methodology

## Selected technique

The selected core technique is the CNN over the outer product of user and movie side-information embeddings. This is identifiable in Figure 1 and Equations 2, 3 and 12 of the paper. The complete comparison adds its flattened global representation and the implicit DeepCF path shown in Figures 2 and 3. Every branch is implemented in this repository rather than imported from CoupledCF code.

## Inputs and representation

Stable sorted original user/movie IDs map to contiguous zero-based indices. User gender, age group, occupation and full ZIP code are categorical fields. ZIP codes remain strings, preserving leading zeros. Each field has a learned eight-dimensional embedding. Their concatenation is projected to an eight-dimensional user side vector. The movie's 18 genre flags form a multi-hot vector projected to an eight-dimensional item side vector. Multiple genres remain present. Vocabulary construction uses metadata only, without interaction targets. This is a transductive catalogue setting.

The paper describes per-field categorical embeddings and concatenation but does not resolve the final dimensions for each field. The equal-length side-vector projection is a disclosed choice. Age is treated as an ordinal category because MovieLens supplies age groups. There are no scalar numeric features requiring normalization in this selection. No review texts or images are required.

## Computation

For side vectors a and b, X[j,k] = a[j] times b[k] is an 8 by 8 outer-product matrix. The local branch applies one valid 3 by 3 convolution with eight output channels, ReLU and 2 by 2 max pooling. The 6 by 6 feature maps become 3 by 3 maps, then flatten to 72 values. The global branch flattens the original X to 64 values. The paper's modeling-settings text on PDF page 10 specifies two CNN layers. Using one convolution is a deliberate reduction for this CPU partial reproduction, rather than a faithful copy of that full network. Padding and pool size are choices not fully specified in the paper.

The implicit branch looks up 32-dimensional user and item ID embeddings p and q. Their elementwise product feeds a dense layer with 64 outputs, batch normalization, ReLU and dropout. The coupled model first concatenates the local 72 values and global 64 values, then transforms those 136 inputs through a 64-unit explicit MLP. It joins that explicit vector with the implicit 64-vector, then applies a second 64-unit fusion layer and scalar output. This preserves the fusion order described in the paper's page 7 text. Hidden layers use ReLU, batch normalization and dropout 0.5. The scorer returns a logit. Binary cross entropy with logits is the numerically stable equivalent of the paper's sigmoid and binary cross entropy. Ranking logits and sigmoid probabilities give the same ordering.

The ablations are `deepcf`, `local`, `global`, `coupled_local` and `coupled`. They omit inactive branches completely. `coupled_local` removes only the global branch from the complete combined model, making that contribution interpretable under a matched budget. Parameter count is measured from trainable tensors for each instantiated model.

## Split and negative sampling

The official archive checksum is verified before preprocessing. Every rating from 1 through 5 denotes a positive interaction. Seed 2026 selects two distinct positives per user without replacement, one validation and one test. All other interactions are training positives. Validation checkpoint selection maximizes NDCG@10, retaining the earliest epoch on a tie.

Each evaluation row has one positive and 99 unique uniformly sampled items outside that user's complete observed-item set. Validation and test negatives are independently drawn and may overlap. Fixed arrays are reused across every seed and model. Four training negatives per positive are resampled each epoch with replacement from unobserved pairs. The sampling seed is `model_seed * 10000 + epoch`, so each matched model uses the same epoch examples and shuffle order. Training negatives may repeat, as expected under sampling with replacement. Held-out positives cannot enter either negative pool or training positives.

The full observed mask protects against false negative labels using retrospective observation membership. It does not feed validation/test preferences into the optimizer. Random holdout can include later events in training than test and cannot support temporal claims. A chronological evaluation would be a separate experiment.

The paper states random leave-one-out testing and mentions random validation splitting without its exact size. This project's additional validation positive per user fixes an otherwise unspecified choice. The catalogue consists of 3,883 metadata items rather than the maximum item-ID value 3,952 reported in the paper. The actual rated-item count is 3,706.

## Metrics and uncertainty

With one positive, HR@K is the fraction of users whose positive rank is at most K. NDCG@K is the mean of 1/log2(rank+1) for hits and zero for misses. There is no additional ideal-DCG factor because it equals one. Exact score ties use increasing encoded item ID, which preserves original ID order and does not favor column zero. Metrics at K=5,10,20 are retained, with K=10 primary. The paper plots K=1 through 10; K=20 is an additional diagnostic.

The aggregate table reports the arithmetic mean and sample standard deviation over model seeds on one fixed split. Paired seed differences and user-activity cohorts provide descriptive analysis. Three seeds and one split do not establish statistical significance or robustness across datasets. A train-only popularity baseline and the exact random-rank expectation provide sanity references.

## Training and evidence

Adam uses learning rate 0.001 and four sampled negatives per training positive. CPU runs use two PyTorch threads and deterministic algorithms. The exact epoch and batch budgets are in `configs/main.json`. CUDA is supported explicitly when installed. The selected state, optimizer state, configuration, selected epoch, validation score and provenance hashes are saved in `best.pt`. Test metrics are computed after checkpoint selection, rather than used to select an epoch.

The paper uses Keras, batch size 256 and a maximum of 100 epochs, with its best MovieLens result around epoch 30. This reproduction uses PyTorch and a bounded CPU budget, with a larger batch. Its optimizer update count and optimization trajectory differ. No hyperparameter sweep is claimed. Timings are observed end-to-end run costs on the recorded machine, including sampling, validation, checkpoint writing and final evaluation. They are not controlled hardware benchmarks or speedup claims. Sampled peak RSS includes the process and loaded data. CUDA memory is null for these CPU experiments.

The first three completed seed-42 variants ran sequentially. Subsequent work uses at most two independent seed processes, each with two PyTorch threads and separate output directories, to fit the available six-core CPU and observed memory use. Concurrent CPU contention and other machine activity can affect recorded wall times. The timing table must not be interpreted as a controlled speed ranking. One preliminary coupled-local run was interrupted after one epoch during a scheduler/source-fidelity correction and is retained under results/interrupted; it is excluded from every reported comparison.

## Reproduction boundary

The software reproduces the selected mathematical mechanism and a matched ablation protocol. It does not replicate every hidden layer, pooling detail, catalogue convention or validation decision from unavailable author code. The review-based Doc2Vec path, additional datasets and rating regression remain outside the experimental scope of this partial reproduction. Published Table 7 is a reference, not an expected target the results are adjusted to reach.
