"""Generate measured tables, plots and controlled ablation differences."""
from pathlib import Path
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coupledcf.data import DataBundle, sha256
from coupledcf.metrics import positive_ranks, ranking_metrics

PAPER = {'deepcf': (.7147, .4312), 'local': (.8212, .5408),
         'global': (.7826, .5252), 'coupled': (.8252, .5440)}


def main():
    config = json.loads((ROOT / 'configs/main.json').read_text())
    data = DataBundle.load(ROOT / config['split_path'])
    split_hash = sha256(ROOT / config['split_path'])
    output = ROOT / 'results'
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    results, history, cohort = [], [], []
    counts = np.bincount(data.train_users, minlength=data.n_users)
    edges = np.quantile(counts, [1 / 3, 2 / 3])
    cohorts = np.digitize(counts, edges)
    for seed in config['seeds']:
        for variant in config['variants']:
            directory = ROOT / config['output_dir'] / f'{variant}_seed{seed}'
            record = json.loads((directory / 'result.json').read_text())
            if record['status'] != 'COMPLETE' or record['split_sha256'] != split_hash:
                raise ValueError(f'Unfinished or mismatched result: {directory}')
            with np.load(directory / 'ranks.npz') as z:
                ranks = z['test']
                if ranking_metrics(ranks) != record['test']:
                    raise ValueError(f'Rank/metric mismatch: {directory}')
            results.append({'variant': variant, 'seed': seed, **record['test'],
                            'selected_epoch': record['selected_epoch'],
                            'completed_epochs': record['completed_epochs'],
                            'parameters': record['parameters'], 'wall_seconds': record['wall_seconds'],
                            'sampled_peak_rss_mb': record['sampled_peak_rss_mb']})
            for row in json.loads((directory / 'history.json').read_text()):
                history.append({'variant': variant, 'seed': seed, 'epoch': row['epoch'],
                                'train_bce': row['train_bce'], 'validation_NDCG@10': row['validation']['NDCG@10']})
            for level, name in enumerate(['lower history', 'middle history', 'higher history']):
                cohort.append({'variant': variant, 'seed': seed, 'cohort': name,
                               'users': int((cohorts == level).sum()),
                               **ranking_metrics(ranks[cohorts == level])})
    frame = pd.DataFrame(results)
    frame.to_csv(output / 'runs.csv', index=False)
    summary = []
    for variant in config['variants']:
        rows = frame[frame.variant == variant]
        record = {'variant': variant, 'runs': len(rows), 'parameters': int(rows.parameters.iloc[0])}
        for key in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20', 'wall_seconds', 'sampled_peak_rss_mb']:
            record[f'{key}_mean'] = rows[key].mean()
            record[f'{key}_std'] = rows[key].std(ddof=1)
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'summary.csv', index=False)
    published_comparison = []
    for variant, (paper_hr, paper_ndcg) in PAPER.items():
        measured = summary[summary.variant == variant].iloc[0]
        published_comparison.append({'variant': variant, 'paper_HR@10': paper_hr,
            'paper_NDCG@10': paper_ndcg, 'measured_HR@10': measured['HR@10_mean'],
            'measured_NDCG@10': measured['NDCG@10_mean'],
            'delta_HR@10': measured['HR@10_mean'] - paper_hr,
            'delta_NDCG@10': measured['NDCG@10_mean'] - paper_ndcg,
            'directly_comparable': False})
    pd.DataFrame(published_comparison).to_csv(output / 'paper_comparison.csv', index=False)
    pd.DataFrame(cohort).to_csv(output / 'cohorts.csv', index=False)
    ablations = []
    for a, b, label in [('coupled', 'deepcf', 'add explicit local and global'),
                         ('coupled', 'coupled_local', 'add global to local plus implicit'),
                         ('coupled_local', 'deepcf', 'add local to implicit'),
                         ('local', 'global', 'local-only versus global-only')]:
        for seed in config['seeds']:
            first = frame[(frame.variant == a) & (frame.seed == seed)].iloc[0]
            second = frame[(frame.variant == b) & (frame.seed == seed)].iloc[0]
            ablations.append({'comparison': label, 'first': a, 'second': b, 'seed': seed,
                              'delta_HR@10': first['HR@10'] - second['HR@10'],
                              'delta_NDCG@10': first['NDCG@10'] - second['NDCG@10']})
    pd.DataFrame(ablations).to_csv(output / 'ablation.csv', index=False)
    popularity = np.bincount(data.train_items, minlength=data.n_items)
    pop_ranks = positive_ranks(popularity[data.test], data.test)
    baseline = {'popularity_train_only': ranking_metrics(pop_ranks),
                'random_rank_expectation': ranking_metrics(np.arange(1, 101)),
                'paper_table7_pdf_page11': {key: {'HR@10': values[0], 'NDCG@10': values[1]}
                                           for key, values in PAPER.items()},
                'split_sha256': split_hash}
    same_genres = np.all(data.item_features[data.test] == data.item_features[data.test[:, :1]], axis=2)
    baseline['feature_diagnostics'] = {
        'distinct_movie_genre_vectors': int(len(np.unique(data.item_features, axis=0))),
        'fraction_users_with_same_genre_negative': float((same_genres.sum(axis=1) > 1).mean()),
        'mean_target_genre_group_size_in_candidates': float(same_genres.sum(axis=1).mean()),
        'interpretation': 'Side-only models receive identical movie features for movies with the same genre flags; this is a representation limitation, not a measured cause of a performance gap.',
    }
    (output / 'references.json').write_text(json.dumps(baseline, indent=2))
    np.save(output / 'popularity_ranks.npy', pop_ranks)
    plt.rcParams.update({'font.size': 16, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for axis, metric in zip(axes, ['HR@10', 'NDCG@10']):
        axis.bar(summary.variant, summary[f'{metric}_mean'], yerr=summary[f'{metric}_std'],
                 capsize=4, color=['#4C657B', '#218D8A', '#6A7285', '#B78346', '#203C66'])
        axis.axhline(baseline['popularity_train_only'][metric], color='#A44F52', linestyle='--', label='Train popularity')
        axis.set_ylabel(metric)
        axis.set_ylim(0, 1)
        axis.tick_params(axis='x', labelrotation=25)
        axis.legend(fontsize=14)
    fig.suptitle('MovieLens 1M sampled ranking, mean and sample SD across seeds')
    fig.tight_layout()
    fig.savefig(figures / 'model_comparison.png', dpi=180)
    plt.close(fig)
    history = pd.DataFrame(history)
    history.to_csv(output / 'learning_curves.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for variant in config['variants']:
        avg = history[history.variant == variant].groupby('epoch').mean(numeric_only=True)
        axes[0].plot(avg.index, avg.train_bce, label=variant)
        axes[1].plot(avg.index, avg['validation_NDCG@10'], label=variant)
    axes[0].set(xlabel='Epoch', ylabel='Training binary cross entropy', title='Mean training loss')
    axes[1].set(xlabel='Epoch', ylabel='Validation NDCG@10', title='Mean validation ranking')
    axes[1].legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / 'learning_curves.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant in ['deepcf', 'coupled']:
        means = frame[frame.variant == variant]
        axis.plot([5, 10, 20], [means[f'HR@{k}'].mean() for k in [5, 10, 20]], marker='o', label=variant)
    axis.set(xlabel='K', ylabel='HR@K', title='Sensitivity to recommendation list length', xticks=[5, 10, 20])
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / 'topk.png', dpi=180)
    plt.close(fig)
    print(summary[['variant', 'runs', 'HR@10_mean', 'NDCG@10_mean', 'parameters']].to_string(index=False))


if __name__ == '__main__':
    main()
