"""Build the academic report from completed, verified experiment records.

Run with the bundled document Python, which supplies ReportLab and pandas.
"""
from pathlib import Path
import csv
import html
import json
import re
from statistics import mean, stdev

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables'


def read_csv(name):
    with (ROOT / 'results' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(exist_ok=True)
    summary = read_csv('summary.csv')
    runs = read_csv('runs.csv')
    ablations = read_csv('ablation.csv')
    config = json.loads((ROOT / 'configs/main.json').read_text())
    student = json.loads((ROOT / 'configs/student.json').read_text())
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    references = json.loads((ROOT / 'results/references.json').read_text())
    by_model = {row['variant']: row for row in summary}
    delta_hr = float(by_model['coupled']['HR@10_mean']) - float(by_model['deepcf']['HR@10_mean'])
    delta_ndcg = float(by_model['coupled']['NDCG@10_mean']) - float(by_model['deepcf']['NDCG@10_mean'])
    direction = 'higher' if delta_ndcg > 0 else 'lower'
    interpretation = (
        'The combined model improves the measured mean ranking score under this budget. '
        'The experiment supports a benefit for this implemented fusion on this split, '
        'but does not isolate all architectural and parameter-count effects.' if delta_ndcg > 0 else
        'The combined model does not improve the measured mean ranking score under this budget. '
        'This negative result does not disprove the paper. Optimization budget, feature encoding, '
        'CNN locality and fusion capacity remain possible causes that this experiment does not isolate.'
    )
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'Times-Bold'
    styles['Title'].fontSize = 27
    styles['Title'].leading = 32
    styles['Title'].spaceAfter = 18
    styles['Heading1'].fontName = 'Times-Bold'
    styles['Heading1'].fontSize = 17
    styles['Heading1'].leading = 21
    styles['Heading1'].spaceBefore = 16
    styles['Heading1'].spaceAfter = 9
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].fontName = 'Times-Bold'
    styles['Heading2'].fontSize = 13
    styles['Heading2'].leading = 16
    styles['Heading2'].spaceBefore = 11
    styles['Heading2'].spaceAfter = 6
    styles['Heading2'].keepWithNext = True
    styles['BodyText'].fontName = 'Times-Roman'
    styles['BodyText'].fontSize = 10.5
    styles['BodyText'].leading = 14.5
    styles['BodyText'].spaceAfter = 7
    styles.add(ParagraphStyle(name='SmallTable', fontName='Times-Roman', fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name='Caption2', fontName='Times-Italic', fontSize=9, leading=12,
                             spaceBefore=5, spaceAfter=10))
    story, markdown = [], []

    def p(text):
        story.append(Paragraph(html.escape(text), styles['BodyText']))
        markdown.append(text + '\n')

    def heading(text, level=1):
        story.append(Paragraph(html.escape(text), styles[f'Heading{level}']))
        markdown.append('#' * (level + 1) + ' ' + text + '\n')

    def table(headers, rows, widths=None):
        values = [[Paragraph(html.escape(str(value)), styles['SmallTable']) for value in row]
                  for row in [headers, *rows]]
        tab = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
        tab.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9EDF2')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#617082')),
            ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#CBD1D8')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        group = [tab]
        if story and isinstance(story[-1], Paragraph) and story[-1].style.name.startswith('Heading'):
            group.insert(0, story.pop())
        story.extend([KeepTogether(group), Spacer(1, 8)])
        markdown.append('| ' + ' | '.join(headers) + ' |')
        markdown.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        markdown.extend('| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows)
        markdown.append('')

    def figure(name, caption, width=6.35 * inch):
        image = Image(str(ROOT / 'results/figures' / name))
        image.drawHeight *= width / image.drawWidth
        image.drawWidth = width
        story.append(KeepTogether([image, Paragraph(html.escape(caption), styles['Caption2'])]))
        markdown.append(f'![{caption}](../results/figures/{name})\n')

    def page():
        # Keep the cover separate; let the main report flow without orphan pages.
        if not any(isinstance(item, PageBreak) for item in story):
            story.append(PageBreak())

    story.append(Paragraph('CoupledCF on MovieLens 1M', styles['Title']))
    markdown.append('# CoupledCF on MovieLens 1M\n')
    p('Research Paper Study and Partial Reproduction')
    p(student['course'] + '. ' + student['institution'] + '. ' + student['group'] + '.')
    table(['Student', 'Roll number', 'Actual contribution'],
          [[s['name'], s['roll_number'], s['contribution']] for s in student['students']], [115, 90, 255])
    heading('Abstract', 2)
    p('This project studies the attribute-based Top-N recommendation mechanism of Zhang et al. [1]. '
      'It implements a convolution over explicit user-item feature interactions, an implicit DeepCF baseline, '
      'and their combination in PyTorch. The study uses the complete official MovieLens 1M interaction set, '
      'one fixed random split, 99 unobserved evaluation candidates per target and matched training procedures. '
      'Five variants are evaluated across three model seeds, with validation-only checkpoint selection.')
    p(f'The complete model achieves mean HR@10 {float(by_model["coupled"]["HR@10_mean"]):.4f} and '
      f'NDCG@10 {float(by_model["coupled"]["NDCG@10_mean"]):.4f}. Its NDCG@10 is {abs(delta_ndcg):.4f} '
      f'{direction} than DeepCF. ' + interpretation)
    p('This is a partial reproduction of the mathematical mechanism, not a claim of exact numerical '
      'replication. The feature projection, catalogue convention, validation holdout and CPU training budget '
      'are explicit adaptations. Human student identities and individual contributions remain editable placeholders.')
    heading('Scope and evidence', 2)
    p('Parts A and B analyze the complete paper. Part C reproduces the attribute-based Top-N component on '
      'MovieLens 1M. All results below come from completed run records, selected checkpoints and per-user ranks. '
      'Rating prediction, review-based embeddings and the other datasets are discussed rather than experimentally reproduced.')
    page()

    heading('Part A Research paper understanding')
    heading('Problem and motivation', 2)
    p('The main implemented problem is personalized Top-N recommendation from implicit feedback. A model '
      'scores an unobserved user-item pair and ranks candidate items. The original paper also considers explicit '
      'rating regression. It is not a sequential recommender, an image-based multimodal model, or a dedicated '
      'explainability method. The authors discuss sparsity and cold start as motivations, but our warm-user '
      'holdout does not test true new-user or new-item generalization.')
    p('User histories are sparse relative to the catalogue. A pure collaborative model learns ID factors '
      'from observed behavior and can miss relationships exposed by demographics or item attributes. '
      'The paper argues that recommendation data are non-independent and heterogeneous. Gender, occupation, '
      'genre and latent preferences can interact across users and items. The practical motivation is to '
      'use these relationships to improve ranked recommendations when behavioral evidence alone is limited.')
    heading('Research gap and claimed contributions', 2)
    p('The authors argue that existing methods do not jointly represent explicit user-item attribute couplings '
      'and implicit latent interactions. Their contribution is a generic neural framework that combines a '
      'CNN-based explicit coupling learner with DeepCF, and separates local CNN patterns from a global '
      'flattened cross-product representation. The framework can receive categorical/numeric side attributes '
      'or review-derived features. These are the authors\' claims, not a proof that all prior recommenders '
      'assume independent data or cannot model feature interactions.')
    p('Related methods in the paper include matrix factorization, NeuMF, Wide and Deep, DeepFM, ConvMF '
      'and A3NCF. NeuMF learns nonlinear collaborative interactions. Wide and Deep combines memorization '
      'with dense generalization. DeepFM combines factorization-machine feature interactions with a deep '
      'network. ConvMF and A3NCF incorporate review-derived information. DeepCF is a paper-internal '
      'ablation and is the appropriate directly implemented baseline here.')
    heading('Paper datasets and evaluation settings', 2)
    p('The original study uses MovieLens 1M and Tafeng for attribute-based Top-K recommendation, '
      'MovieLens 1M and BookCrossing for attribute-based rating prediction, and Amazon Movies and TV '
      'and Yelp for review-based recommendation and rating tasks. Its processed dataset sizes differ by '
      'task and filtering. For MovieLens, Table 2 reports 6,040 users and 1,000,209 ratings. '
      'For Top-K, every observed rating becomes an interaction and four unobserved negatives are sampled '
      'per positive. The evaluation uses one randomly selected observed item and 99 unobserved items per user.')
    p('The paper has reporting ambiguities beyond missing implementation details. Table 3 lists '
      '509,218 Yelp ratings and 192,056 users, averaging about 2.65 ratings per user, while the '
      'text states a minimum of five ratings per user. Those counts and that filter cannot '
      'simultaneously describe the same processed set. Some reported percentage gains also '
      'use absolute metric differences: the MovieLens HR@10 gap of 0.0942 over NeuMF is '
      '9.42 percentage points, or approximately 12.89 percent relative improvement. '
      'This report uses absolute score deltas and verifies its MovieLens counts independently.')
    p('The Top-K settings on PDF pages 9-10 include Adam, learning rate 0.001, ID dimension 32, side '
      'dimension reported as 8 for MovieLens, 3 by 3 filters with eight channels, a 64-unit final hidden '
      'layer, batch size 256 and dropout 0.5. The maximum is 100 epochs, with best performance described '
      'around epoch 30. Table 6 on page 11 concerns rating prediction and is not the Top-K configuration.')
    table(['Published variant', 'HR@10', 'NDCG@10'],
          [['DeepCF', '.7147', '.4312'], ['Local CoupledCF', '.8212', '.5408'],
           ['Global CoupledCF', '.7826', '.5252'], ['CoupledCF', '.8252', '.5440']], [220, 120, 120])
    p('These values are transcribed from Table 7 on supplied PDF page 11, printed page 123954. '
      'They are published references, not measurements produced by this project. The paper reports stronger '
      'MovieLens ranking for the combined method than for its internal DeepCF baseline. It does not provide '
      'the seed-level uncertainty or every implementation detail required to reproduce those values exactly.')
    page()

    heading('Part B Methodology and algorithm analysis')
    p('The model receives user and item identities, four user demographic fields and the movie genre set. '
      'The observable side-information branch and latent ID branch are learned jointly from the binary '
      'interaction objective. A high score means a pair is likely to be an observed interaction relative '
      'to the sampled negatives. It is not a calibrated probability of a future purchase or a star rating.')
    table(['Stage', 'Implementation and output'], [
        ['Input', 'Contiguous user/item IDs; gender, age group, occupation, ZIP; 18 genre flags'],
        ['Feature extraction', 'Four categorical user embeddings; multi-hot movie genre encoding'],
        ['Representation', 'Project side information to user/item vectors of width 8; ID vectors of width 32'],
        ['Explicit interaction', 'Outer product gives 8 by 8 matrix; CNN and max pool give local 72-vector; flatten gives global 64-vector'],
        ['Implicit interaction', 'Elementwise ID embedding product; dense network gives 64-vector'],
        ['Fusion and output', 'Concatenate active branches; dense fusion; one scalar interaction logit; rank items'],
    ], [105, 355])
    heading('Explicit coupling equations', 2)
    p('Let a and b be the learned side vectors. The coupling matrix is X[j,k] = a[j] b[k]. '
      'This outer product includes every cross-coordinate interaction. It is different from the '
      'elementwise product used by DeepCF. The local branch computes ReLU(Conv2D(X)) followed by '
      'max pooling. A 3 by 3 valid filter turns an 8 by 8 matrix into a 6 by 6 map. Eight channels and '
      '2 by 2 max pooling produce eight 3 by 3 maps, hence a local vector with 72 entries. '
      'The global branch preserves all 64 original coupling entries by flattening X.')
    heading('Implicit coupling and prediction', 2)
    p('Let p[u] and q[i] be learned ID embeddings. The implicit interaction is z = p[u] elementwise '
      'times q[i]. A dense layer, batch normalization, ReLU and dropout transform z into a 64-dimensional '
      'representation. The complete model first joins the 72 local and 64 global values and applies '
      'a 64-unit explicit MLP. It concatenates that explicit vector with the implicit 64-vector, '
      'giving 128 joint-fusion inputs. A second 64-unit fusion layer and scalar output produce the ranking logit. '
      'This follows the explicit-MLP-then-implicit-fusion order described on PDF page 7.')
    p('For logit s and binary label y, the objective is binary cross entropy, '
      'L = -y log(sigmoid(s)) - (1-y) log(1-sigmoid(s)). The implementation uses BCEWithLogitsLoss '
      'to compute the same objective stably. Adam optimizes all active branch weights jointly. '
      'Embeddings use a zero-mean normal initialization with standard deviation 0.01. '
      'Dense layers use Xavier initialization and zero biases. Batch normalization and dropout are '
      'active in training and disabled or fixed to their learned statistics in evaluation.')
    heading('The complete paper beyond this implementation', 2)
    p('For review-based side information, the paper builds user and item review documents and learns '
      'Doc2Vec PV-DM representations before coupling them. It discusses Word2Vec and PV-DBOW as '
      'related alternatives. For rating prediction, it changes the output/loss to regression with squared '
      'error and evaluates RMSE and MAE. This project does not substitute genre embeddings for a '
      'claimed Doc2Vec reproduction and does not report rating-regression results.')
    page()

    heading('Part C Implementation and experimental protocol')
    p('The selected technique is implemented directly in src/coupledcf/models.py. data.py owns '
      'source parsing, ID encoding, side features, split construction and negative sampling. '
      'metrics.py owns tie-aware single-positive ranking. engine.py owns Adam training, '
      'validation selection, resource observation and final evaluation. The scripts expose dataset '
      'preparation, training, checkpoint evaluation and result generation. This division keeps '
      'split and negative-pair rules in one place and permits every variant to use the same scorer API.')
    heading('Dataset provenance and preprocessing', 2)
    p('MovieLens 1M was obtained from the official GroupLens archive [2]. Its MD5 is '
      'c4d9eecfca2ab87c1945afe126590906, matching the published checksum. '
      'The parsed data contain 1,000,209 distinct user-item interactions, 6,040 users, 3,883 movie '
      'metadata records and 3,706 rated movies. The maximum original movie ID is 3,952. '
      'The paper lists 3,952 items, so this implementation makes its actual catalogue convention explicit.')
    p('Sorted original IDs define zero-based mappings. ZIP remains a categorical string so leading zeros '
      'are preserved. Gender, age-group, occupation and ZIP vocabularies are recorded in the manifest. '
      'Each user field receives an eight-dimensional embedding and their concatenation is projected to '
      'eight values. Movie genres retain all memberships in an 18-flag multi-hot vector, projected to '
      'eight values. There is no rating threshold: a recorded one-star rating still denotes an observed '
      'interaction for this implicit task. The metadata is available for the full catalogue, so this '
      'is a transductive experiment rather than an unseen-entity deployment test.')
    heading('Matched splits and leakage checks', 2)
    p('Split seed 2026 chooses two distinct random observed items per user, one for validation and one '
      'for test. This leaves 988,129 training positives. Each validation/test target is ranked against '
      '99 unique unobserved items, giving 100 candidates. The exact arrays are reused for all variants '
      'and model seeds. Training samples four unobserved items per positive with replacement each epoch. '
      'Each model with the same seed receives identical sampled examples and shuffle order.')
    p('Checks reject duplicate source pairs, missing metadata references, held-out positives in training, '
      'observed pairs used as negatives, duplicate candidate entries and mismatched split coverage. '
      'The full observed mask excludes held-out positives from negative sampling. Its use is a '
      'retrospective exclusion rule; labels and held-out targets do not enter the optimizer. '
      'The random split does not preserve event time and cannot establish performance on future events.')
    table(['Setting', 'This experiment'], [
        ['Model seeds', ', '.join(map(str, config['seeds']))],
        ['Epochs per variant and seed', str(config['training']['epochs'])],
        ['Training batch / negatives', f'{config["training"]["batch_size"]} pairs / 4 negatives per positive'],
        ['Optimizer / learning rate', 'Adam / 0.001'],
        ['Device / threads', 'CPU / two PyTorch threads'],
        ['Checkpoint selection', 'Highest validation NDCG@10; earliest epoch on ties'],
        ['Primary / secondary metrics', 'HR@10 and NDCG@10 / K=5 and K=20'],
    ], [200, 260])
    heading('Adaptations and controls', 2)
    p('The paper leaves the exact per-field feature dimensions, convolution padding/pooling geometry '
      'and validation allocation incompletely specified. The projections and one validation target '
      'per user above are disclosed choices. PyTorch replaces Keras. The batch is larger and the '
      'training budget is shorter than the paper\'s maximum 100 epochs and its approximate best epoch '
      '30. The paper states two CNN layers; this bounded component reproduction uses one '
      'convolution plus max pooling to reduce CPU computation. This is an architectural adaptation, '
      'not an exact local-network replica. No hyperparameter search is claimed. Every ablation uses identical optimizer settings, '
      'batch size, epochs, split and candidate sets. Model capacity still differs because omitted '
      'branches contain real parameters. These controls support a comparison of the implemented '
      'variants under this budget, rather than an isolated causal claim about CNNs.')
    page()

    heading('Measured model comparison and ablation')
    table(['Variant', 'HR@10 mean +/- SD', 'NDCG@10 mean +/- SD', 'Parameters'], [
        [row['variant'], f'{float(row["HR@10_mean"]):.4f} +/- {float(row["HR@10_std"]):.4f}',
         f'{float(row["NDCG@10_mean"]):.4f} +/- {float(row["NDCG@10_std"]):.4f}', row['parameters']]
        for row in summary], [100, 125, 135, 100])
    p('Completed runs on the fixed MovieLens split. SD is the sample standard deviation '
      'over three model seeds, not a confidence interval over datasets or independently drawn splits. '
      'The DeepCF baseline uses implicit ID interactions. Local and global variants use only side '
      'information. coupled_local joins local and implicit vectors; coupled adds the global vector.')
    figure('model_comparison.png', 'Figure 1. Measured test ranking. Error bars show sample SD over model seeds.')
    pop = references['popularity_train_only']
    rand = references['random_rank_expectation']
    p(f'Train-only popularity achieves HR@10 {pop["HR@10"]:.4f} and NDCG@10 {pop["NDCG@10"]:.4f}. '
      f'The exact uniform-random rank expectation is HR@10 {rand["HR@10"]:.4f} and NDCG@10 '
      f'{rand["NDCG@10"]:.4f} for 100 candidates. These references check the scale of the ranking '
      'scores. A popularity model has no learned personalization, while side-only models can distinguish '
      'users through their attributes.')
    page()
    heading('Controlled comparisons and optimization behavior')
    comparisons = []
    for label in dict.fromkeys(row['comparison'] for row in ablations):
        rows = [row for row in ablations if row['comparison'] == label]
        dh = [float(row['delta_HR@10']) for row in rows]
        dn = [float(row['delta_NDCG@10']) for row in rows]
        comparisons.append([label, f'{mean(dh):+.4f} +/- {stdev(dh):.4f}',
                            f'{mean(dn):+.4f} +/- {stdev(dn):.4f}'])
    table(['First variant minus second', 'HR@10 delta mean +/- SD', 'NDCG@10 delta mean +/- SD'],
          comparisons, [200, 130, 130])
    p(f'The complete model changes mean HR@10 by {delta_hr:+.4f} and mean NDCG@10 by '
      f'{delta_ndcg:+.4f} relative to DeepCF. ' + interpretation)
    heading('Comparison with the published paper', 2)
    comparison = read_csv('paper_comparison.csv')
    table(['Variant', 'Published HR@10 / NDCG@10', 'Measured HR@10 / NDCG@10'],
          [[r['variant'], f'{float(r["paper_HR@10"]):.4f} / {float(r["paper_NDCG@10"]):.4f}',
            f'{float(r["measured_HR@10"]):.4f} / {float(r["measured_NDCG@10"]):.4f}']
           for r in comparison], [110, 175, 175])
    published_delta = float(by_model['coupled']['NDCG@10_mean']) - .5440
    p(f'The combined model\'s mean NDCG@10 differs from published Table 7 by '
      f'{published_delta:+.4f}. This comparison is descriptive rather than a direct replication '
      'score. Catalogue size, fixed candidate draws, added validation holdout, feature encoding, '
      'CNN depth, framework, batch size and optimizer-update budget differ. In particular, ten '
      'epochs at batch 4096 yield far fewer optimizer updates than approximately thirty epochs '
      'at the paper\'s batch 256. The authors\' exact processed splits and complete implementation '
      'details are unavailable in the supplied paper. These differences are plausible explanations '
      'for divergent values, not experimentally isolated causes. Results are preserved even when '
      'they do not reproduce the published ordering or score levels.')
    p('Removing the global branch from the same combined architecture gives a more focused ablation '
      'than comparing only a local-only model with DeepCF. However, adding global inputs also changes '
      'fusion-layer capacity. A parameter-matched control would be needed to distinguish representation '
      'benefits from additional capacity. The local-only versus global-only comparison changes both '
      'interaction processing and optimization, so it is descriptive rather than a pure single-factor test.')
    figure('learning_curves.png', 'Figure 2. Training BCE and validation NDCG@10, averaged across model seeds.')
    selected = [int(row['selected_epoch']) for row in runs]
    p(f'Selected epochs span {min(selected)} through {max(selected)}. '
      f'{sum(epoch == config["training"]["epochs"] for epoch in selected)} of {len(selected)} runs '
      'select the last available epoch. Such runs may remain budget-limited. Decreasing training '
      'BCE alone does not prove that ranked validation quality is improving. Validation scores, '
      'rather than test scores, determine the retained checkpoint. Per-seed histories permit this '
      'claim to be checked without reconstructing a plot by hand.')
    p('The saved cohorts.csv groups users by training-history quantiles and reports each model and '
      'seed separately. Those groups describe variation among warm users. Every MovieLens user '
      'originally has at least 20 interactions, so the lower-history group is not a cold-start cohort. '
      'No unmeasured explanation for a performance gap is treated as established fact.')
    cohort_records = read_csv('cohorts.csv')
    cohort_rows = []
    for name in ['lower history', 'middle history', 'higher history']:
        selected_rows = [r for r in cohort_records if r['cohort'] == name]
        implicit = mean(float(r['NDCG@10']) for r in selected_rows if r['variant'] == 'deepcf')
        coupled = mean(float(r['NDCG@10']) for r in selected_rows if r['variant'] == 'coupled')
        cohort_rows.append([name, selected_rows[0]['users'], f'{implicit:.4f}', f'{coupled:.4f}', f'{coupled-implicit:+.4f}'])
    table(['Warm-user cohort', 'Users', 'DeepCF NDCG@10', 'CoupledCF NDCG@10', 'Difference'],
          cohort_rows, [105, 45, 110, 110, 90])
    p('Cohort scores average the same three model seeds. Groups are defined only by '
      'training-history counts, and tied counts can make their sizes unequal. These comparisons '
      'describe the observed warm-user distribution and are not a cold-start result or an '
      'independent statistical significance test.')
    page()

    heading('Computational requirements and reproducibility')
    table(['Variant', 'Wall seconds mean +/- SD', 'Sampled peak RSS MiB mean'], [
        [r['variant'], f'{float(r["wall_seconds_mean"]):.1f} +/- {float(r["wall_seconds_std"]):.1f}',
         f'{float(r["sampled_peak_rss_mb_mean"]):.1f}'] for r in summary], [120, 180, 160])
    p('Wall time includes sampling, training, validation, checkpoint writes and final evaluation. '
      'The first three seed-42 variants ran sequentially; subsequent independent seeds ran in '
      'at most two processes with two threads each. CPU contention can therefore affect the timings. '
      'These are observed costs on one shared Windows CPU environment, not a controlled speed '
      'comparison or a GPU benchmark. RSS is sampled every quarter second and includes the '
      'Python process, data and libraries. It can miss brief allocation peaks and includes allocator '
      'retention between sequential runs. GPU memory is unmeasured because these runs use CPU PyTorch. '
      'A GPU is supported by the code but was not required to complete these experiments.')
    p('Each epoch processes 4,940,645 labeled examples, comprising 988,129 positive pairs and '
      '3,952,516 sampled negatives. The full experiment has 15 completed runs. Each run records '
      'its actual example and batch counts, selected epoch, train/validation history, test ranks, '
      'parameter count, resource observations, software versions and configuration. A selected '
      'checkpoint includes both model state and optimizer state. Split and checkpoint SHA256 '
      'values identify the exact artifacts. The source files used by the main engine are copied '
      'into each run directory as well as hashed.')
    p('The original paper reports a different Keras/GPU environment. Timing differences from that '
      'study cannot be attributed to the algorithm without matched hardware and implementations. '
      'The local branch performs convolution for each sampled pair and requires more computation '
      'than a simple ID interaction. The measured time table describes that implementation cost '
      'without asserting a general hardware-normalized speed ratio.')
    heading('Software verification', 2)
    p('Automated checks cover known ranking metrics, score ties, non-finite scores, split round trips, '
      'deterministic negatives, held-out leakage rejection, observed-negative rejection, candidate '
      'uniqueness, full-data counts, all model forward/backward paths, explicit branch behavior, '
      'checkpoint round trips and a small completed training run. The full-data pilot additionally '
      'completed one epoch and final test evaluation. scripts/verify_submission.py rechecks the '
      'completed main run records and recomputes predictions from each selected checkpoint. '
      'The notebook execution and artifact checks are recorded in .audit/verification.json.')
    heading('Data retention and reproducibility boundary', 2)
    p('The official archive and extracted files remain local. A redistributable submission excludes '
      'raw MovieLens data, generated data arrays, checkpoints containing feature buffers, environments '
      'and build caches. The full project retains the selected checkpoints locally. The submission '
      'archive includes download/preparation scripts, hashes and measured results. A recipient obtains MovieLens '
      'under its original license. Exact floating-point agreement across different PyTorch versions, '
      'CPU libraries or devices is not guaranteed even with deterministic seeds. The retained '
      'environment versions and source snapshots define the tested configuration.')
    page()

    heading('Critical analysis and proposed improvement')
    heading('Evidence supported strengths', 2)
    p('The project demonstrates that a single modular scorer can learn attribute interactions and '
      'implicit ID preferences jointly, and that each branch can be removed for controlled '
      'comparisons. Side information is compact and available without collecting reviews or images. '
      'The experiment supplies actual ranked evaluation, seed variation, validation selection '
      'and reproducibility records. The measured comparison above, rather than the authors\' '
      'headline gains, determines whether this implementation benefits from explicit coupling.')
    heading('Limitations and generalization', 2)
    p('A CNN assumes meaningful local structure in neighboring coordinates. Learned embedding '
      'coordinates have no fixed semantic spatial order. Convolution can still be a useful '
      'parameter-sharing inductive bias, but the experiment does not establish that adjacent '
      'coordinates represent intrinsically local recommendation relationships. The flattened '
      'global branch may preserve interactions lost by pooling, while also adding fusion parameters.')
    p('Genre-only movie attributes can distinguish content preferences but do not directly encode '
      'movie popularity, quality, cast or text. Demographics are coarse and may introduce unwanted '
      'group differences. ZIP has many categories and can act as a highly specific proxy. '
      'A model using these features would require a separate fairness and privacy assessment '
      'before deployment. No such deployment evaluation is claimed by this mini project.')
    diagnostic = references['feature_diagnostics']
    p(f'The catalogue contains {diagnostic["distinct_movie_genre_vectors"]} distinct genre-flag vectors '
      f'for 3,883 movies. In {100 * diagnostic["fraction_users_with_same_genre_negative"]:.1f}% of '
      'test candidate rows, at least one negative movie has exactly the target movie\'s genre vector. '
      'The side-only models receive the same movie attributes for such movies, while DeepCF can '
      'distinguish their IDs. This measured input limitation is one plausible constraint on '
      'explicit-only performance. It does not isolate the cause of any combined-model gap.')
    p('Random leave-one-out evaluation with sampled negatives is easier than retrieving from a '
      'large full catalogue and may inflate apparent ranking quality. The fixed negative draw '
      'and single split limit robustness. Treating all observed ratings as positive predicts '
      'interaction rather than preference intensity. A user\'s one-star rating and five-star rating '
      'receive the same positive label. That matches the chosen implicit protocol but can '
      'misalign with satisfaction. Other domains may have different side fields, exposure bias '
      'and missing metadata, requiring new encoders and new evaluation.')
    p('True cold start requires evaluating entities whose interaction histories were absent from '
      'training. The side-only branch can, in principle, score a new entity with known attribute '
      'categories, but this implementation fixes catalogue buffers and its combined ID branch '
      'does not learn an unknown-ID policy. This study does not establish cold-start capability. '
      'An explicit unknown-category encoding, held-out-entity split and a side-only fallback '
      'would be required for that claim.')
    heading('A technically meaningful improvement', 2)
    p('A proposed extension replaces convolution over arbitrary latent-coordinate neighborhoods '
      'with field-aware cross attention. Separate user fields and genre tokens would keep their '
      'identities, and learned attention would weight interactions between named fields. A '
      'permutation-invariant aggregation over genre tokens would avoid depending on genre order. '
      'The hypothesis is that this improves generalization and interpretability when side '
      'information is informative. It is not implemented or experimentally validated here.')
    p('To test that hypothesis, retain the current split, candidate sets, seeds, optimizer-update '
      'budget and validation criterion. Match the new model\'s trainable parameter count to '
      'CoupledCF, then add full-catalogue and held-out-user evaluations as separate protocols. '
      'Report all seed outcomes, even if the extension loses. This separates a meaningful '
      'research proposal from an unsupported improvement claim.')
    page()

    heading('Reproduction instructions and conclusion')
    p('Run all commands from the project directory. Create a local virtual environment and install '
      'requirements.txt. Run scripts/download_data.py and scripts/prepare_data.py before the tests. '
      'The main experiment command is python scripts/run_experiments.py --config configs/main.json. '
      'Generate tables with python scripts/summarize_results.py. Recompute one saved model with '
      'python scripts/evaluate.py results/main/coupled_seed42. Execute the notebook with '
      'python scripts/execute_notebook.py, then run python scripts/verify_submission.py. '
      'The README gives exact PowerShell commands and explains fresh output directories.')
    p('A runnable notebook, source code, configurations, tests, checkpoints, result tables, figures, '
      'paper analysis, methodology, presentation, viva answers and a requirement checklist accompany '
      'this report. The editable Markdown report is generated alongside the PDF. Student names '
      'and contribution statements must be filled with verified information before hand-in. '
      'No code in this package can verify an individual\'s academic contribution.')
    p(f'The complete model\'s NDCG@10 is {direction} than DeepCF by {abs(delta_ndcg):.4f} on the '
      'reported fixed split and training budget. ' + interpretation + ' The useful outcome is '
      'a directly implemented coupling mechanism and an auditable comparison, with explicit '
      'limits on what the experiment can establish.')
    heading('References', 2)
    p('[1] Q. Zhang, L. Wang, X. Meng, K. Xu and J. Hu. A Generic Framework for Learning '
      'Explicit and Implicit User-Item Couplings in Recommendation. IEEE Access, vol. 7, '
      'pp. 123944-123958, 2019. DOI: 10.1109/ACCESS.2019.2937841. '
      'Primary source is the supplied 15-page PDF, whose printed pagination is preserved in the analysis.')
    p('[2] GroupLens Research. MovieLens 1M Dataset. Official archive and README. '
      'https://grouplens.org/datasets/movielens/1m/ and '
      'https://files.grouplens.org/datasets/movielens/ml-1m.zip. '
      'Accessed 18 October 2026. The data license remains with the original README.')
    p('[3] F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context. '
      'ACM Transactions on Interactive Intelligent Systems, 5(4), article 19, 2015. '
      'DOI: 10.1145/2827872.')
    p('[4] Research Paper Study and Partial Reproduction: Recommendation Systems. '
      'Supplied four-page assignment brief, including Parts A-C and the 10-mark rubric.')
    p('Source page references use PDF page numbers alongside printed pagination. '
      'Top-K Table 7 is on PDF page 11. Rating prediction Table 6 '
      'on that page is a separate experiment and does not define this reproduction\'s Top-K settings.')
    (ROOT / 'docs/final_report.md').write_text('\n'.join(markdown), encoding='utf-8')

    def footer(canvas, document):
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(48, 28, 'Recommendation Systems partial reproduction')
        canvas.drawRightString(547, 28, str(document.page))

    document = SimpleDocTemplate(str(OUT / 'final_report.pdf'), pagesize=(595.28, 841.89),
                                 leftMargin=48, rightMargin=48, topMargin=45, bottomMargin=48,
                                 title='CoupledCF on MovieLens 1M', author='Student identities to be completed')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Generated docs/final_report.md and deliverables/final_report.pdf')


if __name__ == '__main__':
    main()
