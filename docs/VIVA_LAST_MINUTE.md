# Group 4 — one-page CoupledCF viva revision

**Speaking order by ascending roll number:** I002 Dhruva (slides 1–3) → I012 Viraj (4–7) → I069 Rusheel (8–12) → I077 Shawn (13–16). These are planned presentation assignments, not evidence of previous individual academic contributions.

## What we did in one sentence

We partially reproduced **CoupledCF**, which combines **explicit user/item attribute interactions** with **implicit ID-based user/item interactions**, and evaluated it against DeepCF and explicit-only variants on **MovieLens 1M** using PyTorch.

## Five facts to memorize

1. **Dataset:** 1,000,209 ratings; 6,040 users; 3,883 movie metadata records (3,706 rated movies). The original maximum movie ID is 3,952, not the count of metadata records.
2. **Core math:** User-side 8D vector `a` and movie-side 8D vector `b` make `X = a bᵀ`, an **8×8** explicit coupling matrix; a CNN plus max pooling produces **72** local values and flattening produces **64** global values.
3. **Implicit path:** User ID and item ID have **32D learned embeddings**; their **elementwise product** feeds the implicit DeepCF network.
4. **Evaluation:** One validation and one test positive per user; test target compared with **99 unobserved candidates**; four sampled train negatives per observed positive; metrics are **HR@10** and **NDCG@10**.
5. **Study:** Five variants × three seeds = **15 runs**; **ten CPU epochs** per run. All variants share the same split and ranking candidates.

## Main results (three-seed means)

| Model | HR@10 | NDCG@10 |
| --- | ---: | ---: |
| DeepCF | 0.7825 | 0.5022 |
| Local explicit | 0.3841 | 0.2142 |
| Global explicit | 0.3924 | 0.2151 |
| Local + implicit | 0.7927 | 0.5203 |
| Full CoupledCF | **0.7937** | **0.5224** |

Full versus DeepCF: **+0.0111 HR@10** and **+0.0202 NDCG@10** (absolute score difference). Not a claim of statistical significance.

**HR@10:** the fraction of users whose held-out positive ranks in the first ten.

**NDCG@10:** the mean discounted gain for each user's positive: `1 / log2(rank + 1)` when rank ≤10, otherwise zero. A positive at rank 1 receives more credit than one at rank 10.

## Five examiner traps

1. **Exact reproduction?** No: this is the required partial reproduction. We used one CNN layer rather than the paper's two, feature-projection and protocol adaptations, and only a ten-epoch CPU budget.
2. **Did side information alone work better?** No. Explicit-only branches were substantially worse than DeepCF. The combined models had modestly better ranking results.
3. **Why does the paper score differ?** Different splits/candidate conventions, CNN depth, catalogue, batch size and number of optimizer updates; none individually proven as the cause.
4. **Is cold-start solved?** Not established. We evaluated warm users on a fixed random split, not unseen users or items.
5. **Which is the baseline and ablation?** DeepCF is the implicit-only baseline; remove/add local and global explicit branches to compare. Adding the global branch does **not** consistently improve performance in all three seeds.

## Before presentation

Review `docs/GROUP4_VIVA_PLAYBOOK.md` and `docs/viva_questions.md` for extended speaker scripts and Q&A. Students must personally confirm actual academic contributions and review regenerated report/slides; assigned slides are **not** proof of research/coding contribution.

**Source of measured values:** `results/summary.csv`. **Source of paper methods:** supplied IEEE Access PDF (architecture/equations pp. 4–7, experimental protocol/results pp. 8–13).
