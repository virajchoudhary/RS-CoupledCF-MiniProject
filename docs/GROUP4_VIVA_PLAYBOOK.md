# Group 4: CoupledCF presentation and viva playbook

**Status:** The speaking assignments below are an agreed project plan established for Group 4 preparation, **not** evidence that a particular student personally coded, trained, authored, or tested any model. Real individual contributions must be recorded separately after members verify them.

**Paper:** Q. Zhang et al., *A Generic Framework for Learning Explicit and Implicit User-Item Couplings in Recommendation*, IEEE Access (2019). Use the attached paper's PDF pages 4–7 for architecture and equations and pages 8–13 for datasets, settings and reported results.

**Our scope:** partial reproduction of attribute-based Top-N CoupledCF on MovieLens 1M. Five variants, three random model seeds (42, 43, 44), ten-epoch CPU budget. No claim of exact paper replication or demonstrated cold-start generalization.

## 1. Presentation order and allocated responsibility

| Order | Student | Roll number | Slides | Prepare and present |
|---|---|---|---|---|
| 1 | Dhruva Ambhaikar | I002 | 1–3 | Background, problem, motivation, research gap |
| 2 | Viraj Choudhary | I012 | 4–7 | Paper architecture, explicit/implicit couplings, equations, implemented model |
| 3 | Rusheel Sharma | I069 | 8–12 | MovieLens, splits, negative sampling, evaluation metrics, training and model comparison |
| 4 | Shawn Mendes | I077 | 13–16 | Paper versus experiment, analysis, ablations, limitations and conclusion |

**Rehearsal target:** 12 minutes total, approximately 2 + 3.5 + 3.5 + 3 minutes. If the professor provides a different time limit, shorten the script proportionately. Every member should still understand the whole project, not only their assigned segment.

## 2. The 60-second project summary everyone must know

"Traditional collaborative filtering uses user and item interaction histories. The CoupledCF paper proposes adding explicit relations from user and item attributes, alongside the implicit interactions learned from user/item IDs. We partially reproduced that mechanism in PyTorch on MovieLens 1M. User profile features and movie genre features are projected to vectors, combined using an outer product, and processed through local CNN and global coupling branches. We compared those branches and a combined scorer against DeepCF using the same split, negative candidates, seeds and training budget. Across three seeds, DeepCF achieved HR@10 of 0.7825 and NDCG@10 of 0.5022; the full model achieved 0.7937 and 0.5224. The results show a modest improvement under our specific protocol, but we do not claim exact paper reproduction or statistical significance."

## 3. Speaker 1 — Dhruva (I002), slides 1–3

**Slide 1, introduction:** Introduce the assigned paper and say the goal is research-paper understanding plus *partial reproduction*, not reimplementing every experiment. The main implemented task is personalized Top-N recommendation.

**Slide 2, the problem:** Many users interact with few items relative to the catalogue. The recommender must score a user-item pair and order candidate unseen movies. Define an *interaction* as an observed movie rating (all original values 1–5 are treated as positive interactions for this ranking task).

**Slide 3, research gap:** An ID-only recommender learns latent associations from past interactions but does not explicitly exploit demographic-field/genre relationships. The paper's claim is that local/global explicit coupling may complement learned latent preferences. Distinguish this author claim from what our experiments actually establish.

**Hand-off:** "To explain how both forms of coupling are learned, the next speaker will walk through CoupledCF."

**Likely questions and answers:**

1. **Is the paper about rating prediction or ranking?** Both tasks; our implementation selects attribute-based personalized Top-N ranking.
2. **What is collaborative filtering?** Learning recommendations from patterns of user-item interactions instead of fixed hand-written recommendation rules.
3. **Why use side information?** It may provide additional signals from attributes when interaction-only representations are limited.
4. **Is CoupledCF a sequential recommendation model?** No; the implemented protocol does not model interaction order.
5. **Does the paper prove cold-start performance?** No dedicated cold-start test is in our reproduction; cold start is motivation, not an established project result.

## 4. Speaker 2 — Viraj (I012), slides 4–7

**Slide 4, paper architecture:** CoupledCF merges explicit user-item attribute interaction learning with implicit DeepCF. The explicit path has a local CNN representation and a global coupling representation. The ID-based path uses embeddings and a multilayer perceptron.

**Slide 5, project inputs:** User fields are gender, age group, occupation and ZIP code. Movie features are 18 multi-hot genre flags. Categorical user fields have learned embeddings; these are concatenated and projected to eight-dimensional side vectors. Movie genre flags are projected to eight dimensions. The ID-based user and movie embeddings have 32 dimensions.

**Slide 6, the core mathematical component:** For user-side vector `a` and item-side vector `b`, form the **outer product** `X = a b^T`. With both vectors width 8, `X` is an `8 × 8` matrix containing all pairwise coordinate interactions. Our local path applies one `3 × 3` valid convolution with eight output channels, ReLU and `2 × 2` max pooling. Geometry: `8×8 → 6×6 → 3×3`; flatten eight pooled channels into `8×3×3 = 72` values. The global path flattens the original matrix into 64 values.

**Slide 7, fusion and prediction:** The implicit path forms the elementwise product `p_u ⊙ q_i` of 32D ID vectors, then projects to a 64D representation. Explicit features are fused into 64D; explicit and implicit 64D features are joined and transformed into a scalar logit. Training uses `BCEWithLogitsLoss`; higher logits produce higher recommendation ranks.

**Important fidelity boundary:** The paper states two CNN layers for MovieLens 1M; our partial reproduction uses **one**. Our feature projections and other choices are documented adaptations.

**Hand-off:** "We have seen how the score is computed; next we will explain the dataset and how the models were fairly evaluated."

**Likely questions and answers:**

1. **Outer versus elementwise product?** Outer product produces every pair `a_j b_k` (an 8×8 matrix); elementwise product produces only aligned coordinates (a 32-vector in DeepCF).
2. **Why a CNN?** To learn local patterns among interactions in the constructed feature matrix. Caveat: learned latent coordinate neighborhoods are not inherently semantic.
3. **What is global coupling?** The flattened complete outer-product matrix; unlike the local branch it is not passed through the CNN.
4. **Why use sigmoid/BCE?** Top-N interactions are treated as binary positives versus sampled unobserved negatives. BCEWithLogitsLoss is a numerically stable combined computation.
5. **What is actually implemented?** Our own PyTorch modules under `src/coupledcf/models.py`, not an imported end-to-end CoupledCF library.
6. **What would you change next?** Test field-aware cross-attention with a parameter-matched control; this is a proposal, not a result.

## 5. Speaker 3 — Rusheel (I069), slides 8–12

**Slide 8, data and protocol:** Official MovieLens 1M contains **1,000,209** ratings and **6,040** users. Our encoded metadata catalogue has **3,883** movies; **3,706** are rated. Original maximum movie ID is 3,952, explaining the paper's reported item value versus actual metadata count. For each user, hold out one validation positive and one test positive; **988,129** positives remain for training. Each evaluation target competes against **99 fixed unobserved items**, totaling 100 candidates. Training uses four sampled unobserved negatives per positive.

**Slide 9, model variants:** DeepCF (implicit only), local CNN (explicit only), global (explicit only), local plus implicit, and full CoupledCF (local plus global plus implicit). Fixed dataset split, candidate sets, optimizer settings and seeds across models. Parameter counts differ, so comparisons are controlled in protocol but not capacity-matched.

**Slide 10, outcomes:** See the exact scores in Section 7. State why HR@10 and NDCG@10 complement one another: HR measures whether a hit falls within the first ten; NDCG discounts hits found lower in those ten.

**Slide 11, ablations:** Adding explicit local+global to DeepCF improved mean HR@10 by **0.0111** and NDCG@10 by **0.0202**. However, the incremental effect of *adding global to local+implicit* changed sign across seeds, so we should not say it consistently helps.

**Slide 12, compute:** Runs used CPU PyTorch, ten epochs, batch size 4096, Adam learning rate 0.001. The five variants were each trained for three seeds = 15 runs. Wall times and memory in the report are observed resource use, not controlled speed comparisons.

**Hand-off:** "The measured scores look promising; the final speaker will explain how to interpret them relative to the published paper and what limitations remain."

**Likely questions and answers:**

1. **HR@10 formula?** Number of users with the held-out positive ranked in top ten, divided by evaluated users.
2. **NDCG@10 formula?** For one positive: average `1/log2(rank+1)` if rank ≤ 10, otherwise zero.
3. **Why 99 negatives?** This follows the paper's sampled Top-N ranking approach; full-catalogue retrieval is a different evaluation.
4. **How do you prevent leakage?** Held-out interactions are absent from training, and the complete observed-item mask prevents any known positive being sampled as negative. The mask is used for sampling/exclusion, not optimization labels.
5. **Why train negative sampling?** Interactions are recorded only for observed pairs; sampled unobserved pairs serve as negatives for the binary ranking objective.
6. **Why do all stars count as positives?** This is the selected implicit-interaction ranking protocol; it does not imply a 1-star rating means satisfaction.
7. **What does three seeds prove?** Limited repeatability of optimization on **one** fixed split; not statistical significance or generalizability to new datasets.

## 6. Speaker 4 — Shawn (I077), slides 13–16

**Slide 13, paper reference:** Published paper Table 7 (PDF page 11) reports DeepCF at HR@10 0.7147 / NDCG@10 0.4312; local CoupledCF at 0.8212 / 0.5408; full CoupledCF at 0.8252 / 0.5440. These are author-reported reference scores, **not our measurements**.

**Slide 14, interpretation:** Our full model improves over our DeepCF baseline on all three seeds, but side-only local/global models perform far worse. Integration appears useful for this protocol; explicit information by itself is insufficient under our implementation/training budget. Direct numerical comparison with the paper is limited by different CNN depth, validation split, catalogue convention, projection details, batch size and epoch/update counts.

**Slide 15, limitations and future work:** Single MovieLens dataset; single user-level random split; no true cold start, time-aware split, reviews, rating regression or GPU execution. The authors use two CNN layers but our implementation uses one. We propose, rather than claim to have validated, field-aware cross-attention plus a parameter-matched ablation and alternative dataset.

**Slide 16, team:** State speaking assignments in roll-number order. Do **not** present speaking assignments as already completed individual coding/research contributions. Report actual contributions only after each member confirms them.

**Closing line:** "Our partial reproduction shows a small ranking improvement from combining explicit and implicit coupling under a consistent MovieLens 1M experimental protocol. We preserve both the positive results and the limitations needed for an honest scientific comparison."

**Likely questions and answers:**

1. **Are our scores the same as the paper's?** No. They use different implementation details and training/evaluation conventions.
2. **Can we claim statistical significance?** No; three seeds and one split support descriptive comparisons only.
3. **Why did side-only models underperform?** ID embeddings can memorize interaction-specific preferences in this warm-user setting; attributes alone are coarse. The lower scores are measured; any explanation is a hypothesis.
4. **What happens with new users?** We did not evaluate unseen user IDs. The model uses IDs and transductive metadata; no deployed new-user claim is supported.
5. **Does more training guarantee improved CoupledCF?** No; it changes compute and may improve or hurt results. A controlled experiment is needed.
6. **How would you make the model more interpretable?** Use named user/item fields and field-aware attention, while verifying whether attention provides meaningful explanations rather than assuming it does.

## 7. Final measured result card (three-seed arithmetic means)

| Implementation variant | HR@10 | NDCG@10 | Interpretation |
|---|---:|---:|---|
| DeepCF | 0.7825 | 0.5022 | Implicit ID interaction baseline |
| Local explicit | 0.3841 | 0.2142 | CNN side coupling only |
| Global explicit | 0.3924 | 0.2151 | Flattened side coupling only |
| Local + implicit | 0.7927 | 0.5203 | Local CNN with ID branch |
| **Full CoupledCF** | **0.7937** | **0.5224** | Local + global + ID branches |

Full versus DeepCF: **+0.0111 HR@10** (approximately 1.11 percentage points) and **+0.0202 NDCG@10** (approximately 2.02 points on a 0–1 metric scale).

These numbers come from `results/summary.csv` and the 15 completed `results/main/` run records. They are **not** claims of full reproduction, significance, superiority across datasets or absence of cold-start problems. For per-seed effects see `results/ablation.csv`.

## 8. Questions the examiner may direct to anyone

**Q: What is the difference between explicit and implicit coupling in this paper?**

A: Explicit uses available user/item attributes and their cross-interactions. Implicit uses latent user/item embeddings learned from observed user–item behavior.

**Q: Why not just use matrix factorization?**

A: Standard ID-based factor models do not directly use cross-interactions from the available demographic/genre fields. CoupledCF adds those interactions but at additional model/data complexity.

**Q: What did the ablation teach us?**

A: Combined explicit+implicit learning performed modestly better than our DeepCF baseline; explicit-only variants were much worse. The extra global branch provided inconsistent incremental benefit.

**Q: What were the controls?**

A: Identical MovieLens preparation, held-out users' candidate arrays, sampled negative-generation rules, model seeds, optimizer and epoch budget. Differences in parameter count and architecture remain.

**Q: Did you write the code from the paper or import a finished package?**

A: The selected outer-product coupling, CNN pathway, implicit DeepCF pathway and fusion were implemented in PyTorch in the repository. The project has unit tests and retained run evidence. The group should inspect the code and be prepared to demonstrate it.

**Q: Can your experiments rank every unseen movie?**

A: Not in the reported protocol. It ranks one positive against 99 sampled candidates per user; the model may score other pairs, but full-catalogue retrieval accuracy was not measured.

**Q: How many experiments?**

A: Five variants × three model seeds = 15 main experimental runs; each had a ten-epoch CPU budget. Pilot and interrupted runs were not included in the reported means.

**Q: Why didn't your result match the author's value exactly?**

A: This is a partial reproduction with documented architecture/protocol differences, most notably one CNN rather than two and a much lower optimization-update budget. Do not attribute the difference to a single cause without a controlled experiment.

**Q: What is your individual contribution?**

A: Answer from your **actual** work or review. The speaker allocation alone does not prove who coded, ran experiments, verified artifacts or authored sections. Never make up work history.

## 9. Rehearsal checklist for the group

- Each student can give the 60-second summary without slides.
- Each student can explain their assigned slides in plain language and respond to five questions from their segment.
- Everyone can derive the outer product matrix size (8×8), CNN pooled size (72), and global flatten length (64).
- Everyone knows the dataset counts, held-out test procedure and what HR@10/NDCG@10 mean.
- Everyone remembers DeepCF 0.7825 / 0.5022 and full CoupledCF 0.7937 / 0.5224.
- Everyone can say why published numbers and project measurements cannot be directly equated.
- Each student's real contributions and final hand-in are verified before submission.

## 10. Suggested local demonstration

In the original local project with dataset and selected checkpoints retained:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\evaluate.py results\main\coupled_seed42
.venv\Scripts\python.exe scripts\execute_notebook.py
```

The repo's public ZIP omits MovieLens source data and checkpoints. To demonstrate directly, use the original local workspace; otherwise the README explains complete reproduction from official sources.

**Before any claimed final hand-in:** confirm actual student contributions, rebuild the final PDF, PowerPoint and ZIP, inspect formatting, and perform the institutional submission. See `docs/FINAL_HANDOFF.md`.
