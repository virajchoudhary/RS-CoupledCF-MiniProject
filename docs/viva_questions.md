# Viva questions and answers

## 1. What problem does the implemented model solve?

Personalized Top-N ranking from implicit user-item interactions. It scores candidate items for known users. The original paper also evaluates rating regression, which this project analyzes but does not reproduce.

## 2. What research gap motivates CoupledCF?

The paper aims to learn explicit cross-feature interactions and implicit latent ID interactions jointly. Its CNN learns local patterns in an outer-product matrix, while a separate flattened path retains global interactions.

## 3. What is explicit coupling?

It is the learned interaction between observed side attributes after embedding. In this project user demographics interact with movie genre representations. Explicit coupling does not mean explicit star-rating supervision.

## 4. What is implicit coupling?

It is learned from user and item ID factors and interaction history. DeepCF multiplies the two ID embeddings elementwise and transforms the result through a neural network.

## 5. How does the outer product differ from an elementwise product?

Two length-eight vectors produce 64 cross-coordinate products in an 8 by 8 outer product. An elementwise product produces only eight matching-coordinate products. The explicit branch uses the former, and the implicit branch uses the latter with width 32.

## 6. What is local versus global coupling?

Local coupling applies convolution and max pooling to the coupling matrix. Global coupling flattens the original matrix. The complete model combines both with the implicit vector.

## 7. What shapes pass through the implemented local branch?

The matrix is [batch,1,8,8]. A valid 3 by 3 convolution with eight channels gives [batch,8,6,6]. Max pooling with size two gives [batch,8,3,3], then flattening gives 72 values.

## 8. Is this the exact paper architecture?

No. It reproduces the selected outer-product CNN mechanism and branch structure. The paper states two CNN layers; this bounded partial reproduction uses one. Feature projections, pooling geometry, validation allocation, batch size, framework and epoch budget are also disclosed differences.

## 9. Why use MovieLens 1M?

It is one of the original paper datasets, publicly available from GroupLens, and supplies both interaction history and the demographic/genre side information required for the selected technique.

## 10. How many observations and entities are used?

The source contains 1,000,209 ratings and 6,040 users. Metadata contains 3,883 movies, of which 3,706 have ratings. The maximum movie ID is 3,952, which differs from the count of actual metadata rows.

## 11. Are one-star and two-star ratings negatives?

No. In the chosen paper-style implicit protocol every recorded rating is a positive observed interaction. Negatives are sampled from unobserved user-item pairs. This predicts interaction, not rating intensity.

## 12. How are multiple movie genres encoded?

All 18 genre flags are retained in a multi-hot vector, which is projected to the side-vector width. The code does not select only the first listed genre.

## 13. How are splits and negatives protected from leakage?

Two distinct random observed items per user are held out for validation and test. Training contains all other positives. The complete observed mask excludes every known positive from negative pools. Candidate lists are unique and fixed. Automated checks deliberately insert leakage and require rejection.

## 14. Does the protocol simulate future recommendations?

No. It is random holdout and may train on later events than the test event. A chronological split would test a different deployment question and must be reported separately.

## 15. Why use four training negatives?

The paper uses four uniformly sampled unobserved negatives per positive. This project matches that ratio and resamples each epoch. Sampling with replacement permits repeated negatives.

## 16. Why evaluate against 99 negatives?

The paper ranks one held-out target and 99 sampled unobserved items per user. The total is 100 candidates. Using only nine negatives would make HR@10 trivial and would not match the paper.

## 17. What are HR@10 and NDCG@10?

HR@10 is the fraction of target items ranked in the first ten positions. NDCG@10 rewards those hits by 1/log2(rank+1), so a hit near rank one counts more. With one positive, ideal DCG equals one.

## 18. What happens to exact score ties?

Items with lower original ID rank first. This stable rule is independent of the positive column and is shared across all models.

## 19. Why train with BCEWithLogitsLoss?

It is a numerically stable form of sigmoid plus binary cross entropy, matching the interaction-classification objective. Logit and sigmoid rankings are identical because sigmoid is strictly increasing.

## 20. How is the best checkpoint chosen?

The highest validation NDCG@10 selects the epoch, retaining the earliest epoch on a tie. Test scores do not choose the checkpoint. The saved state is reloaded before final evaluation.

## 21. What makes the comparisons fair?

All variants share the full dataset, split, candidates, seeds, optimizer settings, epoch budget and deterministic per-epoch negative sampling. Parameter counts differ because branches are removed, so capacity is a remaining confounder.

## 22. What do the actual results show?

CoupledCF has mean HR@10 0.7937 and NDCG@10 0.5224. Its NDCG@10 is 0.0202 higher than DeepCF. Read results/summary.csv for all five variants and sample standard deviations, and results/ablation.csv for paired differences.

## 23. Why might the combined model underperform?

Possible causes include a shorter optimization budget, larger batches, coarse genre information, arbitrary CNN coordinate locality, dropout and fusion optimization. The measured ablations and curves describe behavior, but they do not prove which of these hypotheses caused a gap.

## 24. Does a larger score than the paper prove a better implementation?

No. Catalogue, split, candidate draws, training updates and architecture differ. Published Table 7 is a reference under a different experimental setup, not an apples-to-apples ranking of implementations.

## 25. What does three-seed standard deviation mean?

It describes model-seed variation on one fixed split. It is not uncertainty over datasets or split choices, and it does not by itself establish statistical significance.

## 26. Is cold start solved?

No cold-start experiment was run. All evaluated users have training histories. Side features may help new entities in principle, but the combined model has fixed ID tables and no validated unknown-ID policy.

## 27. How does the review-based path in the paper differ?

It aggregates reviews into user and item documents and uses Doc2Vec PV-DM vectors as side information. This project uses metadata and does not claim to have trained Doc2Vec.

## 28. What compute is needed?

The completed study uses CPU PyTorch with two threads. CUDA is supported in the pipeline when available but was not executed. Actual parameter counts, run times and sampled process RSS are saved for every run. These timings are observations, not controlled hardware speedup comparisons.

## 29. What improvement is proposed?

Field-aware cross attention could model named demographic fields and genre tokens without assuming local neighborhoods in arbitrary embedding coordinates. A parameter-matched evaluation with the same split/seeds and additional cold-start protocols would test the hypothesis. It is not a measured improvement.

## 30. How can another person reproduce a reported number?

Download and verify the official archive, prepare the split, create the documented environment, then run the fixed config or evaluate a retained checkpoint. Hashes, source snapshots, configuration, histories and per-user ranks identify every completed result.

## 31. What did each student contribute?

Replace the placeholders in configs/student.json with the actual names, roll numbers and work each person can explain and defend. Do not invent individual contributions or claim experimental work that was not performed.

