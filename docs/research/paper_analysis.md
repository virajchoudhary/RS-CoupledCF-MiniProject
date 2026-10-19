# Paper analysis: CoupledCF

## Source and scope

This analysis is based on the supplied 15-page scan of Q. Zhang et al.,
"A Generic Framework for Learning Explicit and Implicit User-Item Couplings in
Recommendation," *IEEE Access*, vol. 7, 2019, DOI
10.1109/ACCESS.2019.2937841. The PDF is an image-only scan. Page references
below use its PDF page numbers, not the printed IEEE page numbers. The diagrams,
equations, tables, and figure captions were inspected in rendered page images.

The paper studies two personalized recommendation tasks: implicit Top-K item
ranking and explicit rating prediction. It is not a sequential,
explainable, or multimodal recommender. Its central claim is that a recommender
can jointly learn latent ID-based user-item interactions and interactions
between user and item side-information features. The authors call the resulting
model CoupledCF. [PDF pp. 1-3, 14]

## Problem, motivation, and gap

The paper targets collaborative filtering when ratings and side information
are sparse. ID-based methods learn user and item latent factors, but the paper
argues that they treat the users and items as independent ID entities and do
not model interactions among their attributes. This can underuse demographics,
item attributes, and review-derived representations. The authors also point
out that models which add side data often learn user or item representations
without jointly modeling explicit feature interactions and implicit
interaction history. [PDF pp. 2-3]

The authors frame their contribution as a general architecture for learning
local and global explicit user-item couplings from side information, then
combining those couplings with an implicit collaborative-filtering model. They
claim this can improve recommendation accuracy and help with sparsity and
cold-start cases. Their experiments compare ranking and rating prediction on
several datasets, including attribute-based and review-based features.
[PDF pp. 1-3, 8-14]

The paper claims four contributions. It introduces a CNN-based local coupling
model and a global coupling model based on the flattened coupling matrix. It
combines those explicit representations with DeepCF's implicit user-item
interactions. It applies the framework to both attribute-based features and
review-derived user and item representations. It evaluates four CoupledCF
variants against task-specific baselines on five datasets. The cold-start
claim is not tested with a user-held-out or item-held-out experiment.
[PDF pp. 2-3, 8-14]

The paper's experiments measure random train, validation, and test splits.
They do not report a separate cold-start split, a chronological split, or a
new-user/new-item holdout protocol. The cold-start statement is therefore a
motivation and claimed applicability, not an experimentally isolated result.
[PDF pp. 3, 9-10, 14]

## Inputs and task outputs

The model has two kinds of input:

| Input | Attribute-based datasets | Review-based datasets |
|---|---|---|
| User and item identity | User ID and item ID, used by DeepCF's embedding tables | User ID and item ID, used by DeepCF's embedding tables |
| User side information | Demographic fields such as gender, age, occupation, region, or postal code | A document embedding built from reviews written by that user |
| Item side information | Fields such as movie genres or book and product attributes | A document embedding built from reviews written about that item |
| Training target | Binary user-item interaction for Top-K, or observed rating for rating prediction | Binary user-item interaction for Top-K, or observed rating for rating prediction |

The attribute examples and review preprocessing are described on PDF pp. 7-8.
The Top-K task predicts an interaction probability and ranks candidate items.
The rating task predicts a scalar rating. The paper uses a sigmoid output for
Top-K and a linear output for rating regression. [PDF pp. 6-8]

## Architecture

Figure 1 presents the explicit local and global coupling branches. Figure 3
shows how CoupledCF combines that explicit representation with DeepCF. The
equations and text on PDF pp. 4-7 define the following computation.

### Side-feature representations

For an entity with categorical attributes, the paper represents each category
as a one-hot vector and learns a dense embedding from it. For a categorical
field \(x_i\), equation 17 is \(c_i = W_i^T o_i\), where \(o_i\) is its one-hot
vector and \(W_i\) is the learned embedding matrix. Numeric fields are
normalized. The categorical embeddings and normalized numeric fields are
combined into a user or item feature vector. [PDF p. 7, Eq. 17, Fig. 4]

For review-based features, the paper uses Doc2Vec's PV-DM model to learn one
document vector for each user and one for each item. The user document combines
reviews written by that user. The item document combines reviews written
about that item. This is review-derived side information; the model does not
feed raw review tokens through the CoupledCF network. [PDF pp. 7-8]

### Explicit local coupling

Let \(u_c\) and \(v_c\) be the user and item feature vectors. Equation 12
defines their coupling matrix \(X_c\) as their outer product:

\[
X_c = u_c v_c^T
\]

Each matrix entry corresponds to an interaction between one user feature and
one item feature. The paper applies a CNN to \(X_c\) to learn local feature
interactions. Equation 2 gives a convolution followed by an activation,
\(a_c = g(W * X_c + b)\). A pooling function then reduces the feature map,
with max pooling described in equation 3. The pooled result is flattened into
the local coupling vector \(e_l\). The authors use ReLU for the CNN activation
in CoupledCF. [PDF pp. 5-6, Eqs. 2-3, Fig. 1]

### Explicit global coupling

The global branch flattens \(X_c\) directly into the global coupling vector
\(e_g\). Figure 1 and the text on PDF pp. 4-6 distinguish this path from the
convolution and pooling path. The global vector does not pass through the
local CNN. The paper does not describe another CNN or projection on this
branch. [PDF pp. 4-6, Fig. 1]

### Implicit DeepCF coupling

DeepCF maps one-hot user and item IDs to latent vectors \(p\) and \(q\) using
learned embedding matrices (equations 4 and 5). It forms the elementwise
product \(p \odot q\), which captures latent user-item interactions. An MLP
then learns higher-order interactions. Equation 6 defines hidden layers as
affine transformations followed by ReLU. For Top-K prediction, the final
output is passed through a sigmoid to estimate an interaction probability.
For a rating task, the last sigmoid is omitted and the final dense layer
returns a rating estimate. [PDF pp. 5-7, Eqs. 4-8, Fig. 2]

### CoupledCF fusion and variants

CoupledCF first concatenates the local vector \(e_l\) and global vector
\(e_g\), then uses an MLP to learn an abstract explicit coupling
representation. It concatenates that representation with DeepCF's implicit
output and passes the result through a fully connected layer. For Top-K, the
final output is a sigmoid probability. For rating prediction, the final
output is linear. Equations 12-16 and Figure 3 show this combined flow.
[PDF pp. 6-7, Eqs. 12-16, Fig. 3]

The paper evaluates four variants:

| Variant | Learned representation |
|---|---|
| DeepCF | Implicit ID-based interactions through latent user and item embeddings |
| lCoupledCF | Local explicit side-feature interactions learned by the CNN |
| gCoupledCF | Global explicit side-feature interactions learned from flattened \(X_c\), without the local CNN path |
| CoupledCF | Local explicit, global explicit, and implicit DeepCF representations |

The variant definitions appear on PDF pp. 8-9. This comparison is the paper's
main component ablation.

## Learning objectives

For Top-K recommendation, the paper converts observed user-item ratings into
binary interactions. It samples negative user-item pairs from unobserved
interactions and trains the model as a binary classifier. Equation 9 is binary
cross-entropy on the predicted interaction probability. Equations 7-8 define
the probability output. [PDF pp. 6, 8, Eqs. 7-9]

For rating prediction, the model outputs a real-valued score and uses squared
error as the per-example loss. Equation 10 gives the rating loss, and equation
11 averages the per-example loss over the training set. The paper uses Adam
and reports task-specific learning rates. [PDF pp. 6, 10, Eqs. 10-11]

These objectives are separate experimental tasks. A reproduction should not
use rating loss for the binary Top-K runs or apply the Top-K sigmoid to rating
predictions.

## Data and preprocessing

The paper reports five unique datasets. MovieLens 1M and Ta-Feng support
attribute-based Top-K experiments. MovieLens 1M and BookCrossing support
attribute-based rating prediction. Amazon Movies and TV (AMT) and Yelp support
review-based Top-K and rating prediction. [PDF pp. 1, 8-9, 14]

| Dataset | Users | Items | Interactions or ratings | Sparsity |
|---|---:|---:|---:|---:|
| MovieLens 1M | 6,040 | 3,952 | 1,000,209 | 95.81% |
| Ta-Feng | 32,266 | 23,812 | 817,741 | 99.89% |
| BookCrossing after filtering | 3,704 | 10,000 | 133,699 | 99.64% |
| AMT after review preprocessing | 23,982 | 98,593 | 217,568 | 99.99% |
| Yelp after review preprocessing | 192,056 | 139,188 | 509,218 | 99.99% |

These values are from Tables 2 and 3 on PDF p. 8. MovieLens 1M user features
include gender, age, occupation, and postal code; item features include movie
genres. Ta-Feng user attributes are customer ID, age, and region; item
attributes include subclass, amount, and price. BookCrossing uses user location and age and book title, author,
publication year, and publisher. [PDF p. 8]

For BookCrossing, the authors remove zero ratings and filter users using the
same minimum of 20 user-item interactions as MovieLens 1M. The filtered data
contains 133,699 ratings, 3,704 users, and 10,000 books. For AMT and Yelp,
they extract user ID,
item ID, rating, and review text; combine reviews by user and by item into
documents; remove punctuation, stopwords, and terms with frequency below 10;
retain user-item reviews with at least 50 words; and require at least five
ratings per user. The paper reports the post-filtering counts above. [PDF p. 8]

For Top-K, the paper binarizes observed ratings and uniformly samples four
unobserved user-item pairs for each positive training instance. These are
training negatives, not the candidate negatives used for final ranking.
[PDF p. 8]

## Splits, evaluation, and hyperparameters

### Data split and ranking protocol

The authors randomly divide each dataset into training, validation, and test
sets, then tune CoupledCF and baselines on validation data. They do not state
the split proportions, a random seed, or whether the random split preserves
time order. [PDF p. 9]

For Top-K evaluation, the paper holds out one interacted item for each test
user and samples 99 unobserved items to form a 100-item candidate set. Models
rank those 100 items. The paper reports Hit Ratio and Normalized Discounted
Cumulative Gain at K values from 1 to 10, with the main tables reporting
HR@10 and NDCG@10. Equations 19 and 20 define the metrics. [PDF p. 10,
Eqs. 19-20]

### Top-K settings

The following are the reported Top-K settings on PDF pp. 9-10. They must be
kept distinct from Table 6's rating-prediction settings on PDF p. 11.

| Setting | Reported choice |
|---|---|
| DeepCF ID embedding dimension | Searched 8, 16, 32, and 64; selected 32 for all four Top-K datasets |
| Explicit CNN side embedding dimension | Searched 8, 16, 32, 64, 128, and 256; selected 8 for MovieLens 1M and Ta-Feng, 64 for AMT and Yelp |
| CNN | Two convolutional layers, 3 by 3 filters; 8 channels for MovieLens 1M and Ta-Feng, 64 for AMT and Yelp |
| CoupledCF hidden representation width | Searched 8, 16, 32, 64, 128, and 256; the paper describes 64 fully connected units for MovieLens 1M and Ta-Feng and 32 for AMT and Yelp |
| Learning rate | 0.001 for MovieLens 1M, 0.005 for Ta-Feng, 0.00001 for AMT and Yelp |
| Batch size | Searched 32, 64, 128, and 256; selected 256 |
| Epochs | 100 maximum; the paper reports best performance around epoch 30 for MovieLens 1M and Ta-Feng and around epoch 10 for AMT and Yelp |
| Regularization and initialization | Batch normalization, dropout 0.5; ID embeddings initialized from a normal distribution with mean 0 and standard deviation 0.01; dense-layer weights use Glorot uniform initialization and biases are zero |
| Activations and optimizer | ReLU in hidden dense layers; sigmoid final output for DeepCF and CoupledCF Top-K prediction; Adam |
| Doc2Vec | PV-DM, vector size 100, epochs 100, window size 5, minimum count 10 |

The article sometimes calls a representation dimension a number of
"embedding layers" or "fully connected layers." The table above records the
values as dimensions or widths because the surrounding text describes a
dimension sweep; this wording ambiguity should not be silently turned into a
claim about a network's layer count. [PDF pp. 9-10]

### Rating-prediction settings

Table 6 on PDF p. 11 is explicitly titled as the optimal settings for rating
prediction. These values do not define the Top-K model configuration.

| Setting | MovieLens 1M | BookCrossing | AMT | Yelp |
|---|---:|---:|---:|---:|
| DeepCF ID embedding dimension | 32 | 32 | 32 | 32 |
| Explicit CNN embedding dimension | 16 | 8 | 32 | 64 |
| CNN filter size and channels | (3, 3), 16 | (3, 3), 8 | (3, 3), 32 | (3, 3), 64 |
| Learning rate | 0.0001 | 0.0001 | 0.00001 | 0.00001 |
| Epochs | 30 | 50 | 20 | 20 |
| Batch size | 128 | 128 | 128 | 128 |

The paper says these settings were selected by validation loss. It reports a
3.5 GHz machine, an NVIDIA GeForce 1080Ti, and 32 GB of memory, but it does not
make clear whether the 32 GB refers to system RAM or GPU memory. It reports no
training-time measurements. [PDF pp. 9, 11]

## Baselines

The paper compares its variants against methods selected by task and feature
type. Attribute-based Top-K baselines are NeuMF, Wide&Deep, and DeepFM.
Review-based Top-K baselines are NeuMF, ConvMF, and A^3NCF. Attribute-based
rating baselines include BMF and DeepFM; review-based rating baselines include
BMF and ConvMF. Tables 4, 7, 8, 9, and 10 list the comparisons. The methods are
not all used on every dataset. [PDF p. 9, Tables 4 and 7-10]

## Reported results

The values below are transcribed from the paper's result tables. They are
published values, not results from this project. HR and NDCG values are
reported at K=10.

### Attribute-based Top-K

Table 7 reports MovieLens 1M and Ta-Feng:

| Model | MovieLens HR@10 | MovieLens NDCG@10 | Ta-Feng HR@10 | Ta-Feng NDCG@10 |
|---|---:|---:|---:|---:|
| NeuMF | 0.7310 | 0.4480 | 0.6519 | 0.4329 |
| Wide&Deep | 0.7300 | 0.4470 | 0.6420 | 0.4522 |
| DeepFM | 0.6452 | 0.3703 | 0.6821 | 0.4522 |
| DeepCF | 0.7147 | 0.4312 | 0.6506 | 0.4322 |
| lCoupledCF | 0.8212 | 0.5408 | 0.6798 | **0.4700** |
| gCoupledCF | 0.7826 | 0.5252 | 0.6643 | 0.4205 |
| CoupledCF | **0.8252** | **0.5440** | **0.6953** | 0.4623 |

CoupledCF has the highest MovieLens 1M HR@10 and NDCG@10 and Ta-Feng HR@10 in
Table 7. lCoupledCF has the highest Ta-Feng NDCG@10. Thus, the full model does
not lead every reported metric. [PDF p. 11, Table 7]

### Review-based Top-K

Table 8 reports AMT and Yelp:

| Model | AMT HR@10 | AMT NDCG@10 | Yelp HR@10 | Yelp NDCG@10 |
|---|---:|---:|---:|---:|
| NeuMF | 0.4579 | 0.2554 | 0.4420 | 0.2635 |
| A^3NCF | 0.4609 | 0.2563 | 0.4217 | 0.2581 |
| ConvMF | 0.4848 | 0.2689 | 0.4756 | 0.2793 |
| DeepCF | 0.4308 | 0.2356 | 0.4545 | 0.2512 |
| lCoupledCF | 0.5987 | 0.3423 | 0.7564 | 0.5421 |
| gCoupledCF | 0.4876 | 0.2975 | 0.4680 | 0.3429 |
| CoupledCF | **0.6017** | **0.3567** | **0.7888** | **0.5629** |

CoupledCF leads all four AMT and Yelp measures in Table 8. The result supports
the paper's claim that review-derived user and item feature representations
can help these experiments. The table does not establish that the same gain
holds under different splits or for new users and items. [PDF p. 11, Table 8]

### Rating prediction

Table 9 reports MovieLens 1M and BookCrossing:

| Model | MovieLens RMSE | MovieLens MAE | BookCrossing RMSE | BookCrossing MAE |
|---|---:|---:|---:|---:|
| BMF | 1.1334 | 0.7929 | 0.9200 | 0.5215 |
| DeepFM | 0.9504 | 0.7452 | 0.8437 | 0.4104 |
| DeepCF | 0.9304 | **0.7264** | 0.8013 | 0.2705 |
| lCoupledCF | 1.1484 | 0.9462 | 0.9450 | 0.5752 |
| gCoupledCF | 0.9406 | 0.7467 | 0.8064 | 0.4074 |
| CoupledCF | **0.9295** | 0.7285 | **0.5922** | **0.2198** |

Table 10 reports AMT and Yelp:

| Model | AMT RMSE | AMT MAE | Yelp RMSE | Yelp MAE |
|---|---:|---:|---:|---:|
| BMF | 1.1290 | 0.7929 | 1.5452 | 1.3580 |
| ConvMF | 0.9879 | 0.7233 | 1.3191 | 1.0237 |
| DeepCF | 1.1045 | 0.7632 | 1.2963 | 1.0095 |
| lCoupledCF | 1.0057 | 0.7324 | 1.2630 | 0.9943 |
| gCoupledCF | 1.0109 | 0.7425 | **1.2479** | 0.9897 |
| CoupledCF | 0.9610 | **0.6636** | 1.2598 | **0.9678** |

Lower RMSE and MAE are better. The full model is not best on every rating
metric: DeepCF has lower MovieLens 1M MAE, and gCoupledCF has lower Yelp RMSE.
CoupledCF has the best MovieLens 1M RMSE, both BookCrossing scores, AMT RMSE
and MAE, and Yelp MAE in these tables. [PDF p. 13, Tables 9-10]

The authors also show performance against epoch count and include two
illustrative recommendation lists. Their text reports that Top-K performance
peaks around epoch 30 for MovieLens 1M and Ta-Feng and around epoch 10 for AMT
and Yelp; they warn that later epochs can overfit. [PDF pp. 12-13, Figs. 9-10]

## Critical assessment

### Strengths

- The model makes the interaction structure explicit. The outer product
  distinguishes interactions between side-feature dimensions, while the CNN
  and max pooling learn local patterns and the flattened branch learns a
  separate global representation. [PDF pp. 4-7]
- The model joins side information and ID-based collaborative filtering in a
  single prediction network. This permits separate local, global, implicit,
  and combined comparisons. [PDF pp. 6-9]
- The paper evaluates both ranking and rating prediction and uses both
  attribute-based and review-based side information. [PDF pp. 8-13]

### Limitations and reproducibility

- Table 3's Yelp counts and the stated user filter are internally inconsistent.
  The paper says each user has at least five ratings, but 509,218 ratings for
  192,056 users average about 2.65 ratings per user. A five-rating minimum
  would require at least 960,280 ratings. The paper does not resolve whether
  the user count or the filtering description refers to a different stage.
  These are source-reporting uncertainties, not counts adopted by this
  MovieLens reproduction. [PDF p. 8, Table 3]
- Some stated percentage improvements are absolute metric differences
  expressed as percentages. For example, the MovieLens HR@10 difference
  between CoupledCF and NeuMF is 0.8252 - 0.7310 = 0.0942, which is 9.42
  percentage points and approximately 12.89 percent relative to NeuMF. The
  paper's text calls this 9.42 percent. This project reports absolute metric
  deltas explicitly rather than treating those definitions as interchangeable.
  [PDF p. 11, Table 7 and accompanying text]
- The paper omits split proportions and random seeds. It describes random
  train, validation, and test sets but does not say whether they are
  chronological. Its one-held-out-item evaluation can produce results that
  differ from a time-aware recommendation task. [PDF pp. 9-10]
- It gives no uncertainty estimates, repeated-seed results, or significance
  tests. Small table differences, such as the MovieLens Top-K gap between
  CoupledCF and lCoupledCF, may not be stable across splits. [PDF pp. 11-13]
- The article does not report all dimensions of the MLP that combines local
  and global vectors. Some hyperparameter descriptions say "embedding layers"
  or "fully connected layers" when the context appears to describe dimensions.
  The global vector itself is the flattened coupling matrix. [PDF pp. 4-7,
  9-11]
- The paper describes negative sampling for training and candidate sampling
  for evaluation, but leaves implementation details such as random seeds and
  repeated-sampling policy unreported. It also does not fully specify all
  data filtering and ID mapping steps needed to recreate the tables. [PDF
  pp. 8-10]
- The article reports a GPU and memory configuration but no wall-clock
  training time, inference latency, parameter count, or memory use for the
  outer-product branch. The feature-interaction matrix grows with the product
  of user and item feature-vector widths. [PDF pp. 4-5, 9]
- The ID-based branch still requires known user and item identities.
  Side-feature learning could help with sparse histories, but the paper does
  not show a controlled cold-start experiment for unseen users or items.
  [PDF pp. 5-7, 14]
- Results are not uniform across variants or metrics. For example,
  lCoupledCF has the best Ta-Feng NDCG@10, DeepCF has the best MovieLens MAE,
  and non-combined baselines lead selected rating metrics. The evidence
  supports gains in many settings, not universal dominance. [PDF pp. 11-13]

The article says the authors implemented CoupledCF in Python using Keras and
that the baselines follow their GitHub experiment configurations. The scanned
article does not identify a CoupledCF repository, release an exact split, or
provide checkpoints. An exact numerical reproduction is therefore not
specified by the paper alone. [PDF p. 9]

### Proposed improvement

The project's primary proposed extension is field-aware cross attention over
named user fields and genre tokens, with a permutation-invariant aggregation
over movie genres. It would avoid assuming spatial neighborhoods among
arbitrary embedding coordinates. Match trainable parameter counts, split,
candidates, seeds and optimizer-update budget against CoupledCF before
attributing a benefit to the interaction mechanism. This is an unimplemented,
untested hypothesis and is the same proposal used in the report and slides.

Evaluate user-cold-start and item-cold-start cases separately as additional
validation protocols. Hold out users or items from training, retain side
information, and compare with a feature-only path that does not require their
ID embeddings. Those experiments would test the cold-start motivation
directly. They are proposed checks rather than results from the paper or this
reproduction.

## Journal extension and future work

Section VI and Table 13 explain that the journal paper extends the IJCAI 2018
CoupledCF conference paper. The added material includes review-based side
information through Doc2Vec, rating prediction, more datasets and comparison
methods, and more experimental analysis. The conference version focuses on
attribute-based Top-K recommendation on MovieLens 1M and Ta-Feng. [PDF p. 14]

The conclusion proposes richer real-world user/item information, more effective
coupling functions, social relationships, better embeddings and multimedia
information such as images and videos. These are future directions in the
paper, rather than implemented experimental inputs. The illustrative
recommendation lists in Tables 11 and 12 are qualitative examples, not an
additional controlled evaluation or an explanation guarantee. [PDF pp. 13-14]

## Reproduction facts and unresolved details

| Topic | What the article states | What remains unreported or ambiguous |
|---|---|---|
| Problem | Top-K implicit interaction ranking and rating prediction | No sequential evaluation or separate cold-start protocol |
| Side information | Demographics, item attributes, and Doc2Vec review representations | Some field encoding and normalization details |
| Explicit coupling | Outer product, local CNN plus pooling, and flattened global vector; local and global vectors feed an MLP | Full explicit-fusion MLP widths and some convolution settings |
| Implicit coupling | User and item ID embeddings, elementwise product, MLP | Full DeepCF layer widths by experiment |
| Top-K negatives | Four uniform unobserved training pairs per positive; 99 sampled unobserved items for each held-out test positive | Seed, replacement policy, and exact candidate sampling code |
| Data split | Random train, validation, and test sets; hyperparameter tuning on validation | Split proportions, seed, temporal ordering |
| Top-K hyperparameters | See PDF pp. 9-10 and the table above | Ambiguous wording for some dimension-versus-layer-count descriptions |
| Rating hyperparameters | Table 6, PDF p. 11 | Table 6 applies to rating prediction, not Top-K |
| Metrics | HR@K and NDCG@K for Top-K; RMSE and MAE for ratings | Repeated runs and uncertainty estimates |
| Compute | NVIDIA GeForce 1080Ti and 32 GB memory reported | Whether 32 GB is host RAM or GPU memory; no runtime or parameter count |
| Code and artifacts | Keras implementation and baseline GitHub experiment configurations mentioned | No exact source repository, processed splits, or checkpoints identified |

The design-only module and tensor contracts for a minimal implementation are
in [the CoupledCF implementation sketch](../../.audit/design_a/design.md).
