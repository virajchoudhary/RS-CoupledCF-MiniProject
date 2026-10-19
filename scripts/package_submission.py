"""Package source and measured deliverables without the licensed dataset."""
from pathlib import Path
import hashlib
import json
import zipfile

from final_preflight import validate_final

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'deliverables/submission.zip'


def main():
    # Do not distribute a supposedly final package with placeholders or stale exports.
    validate_final(ROOT)
    files = []
    for name in ['README.md', 'requirements.txt', 'requirements-lock.txt', 'requirements-artifacts.txt',
                 'pyproject.toml', 'decisions.tsv',
                 'Group 4_generic frame work for explicit and implicit.pdf',
                 'Research Paper Study and Partial Implementation.pdf']:
        files.append(ROOT / name)
    for directory in ['src', 'scripts', 'tests', 'configs', 'docs', 'notebooks', 'results']:
        for path in (ROOT / directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pt'):
                continue
            if directory == 'results' and any(p in path.parts for p in ('pilot', 'interrupted')):
                continue
            files.append(path)
    for name in ['final_report.pdf', 'CoupledCF_presentation.pptx', 'verification.json', 'submission_guide.md']:
        files.append(ROOT / 'deliverables' / name)
    provenance = {
        'split_sha256': json.loads((ROOT / 'data/processed/manifest.json').read_text())['split_sha256'],
        'source_url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'archive_md5': 'c4d9eecfca2ab87c1945afe126590906',
        'excluded': ['raw and processed MovieLens data', 'checkpoints containing MovieLens feature buffers',
                     'virtual environment', 'private build caches', 'pilot and interrupted runs'],
        'local_checkpoints': 'The full workspace retains results/main/*/best.pt for direct evaluation.',
    }
    with zipfile.ZipFile(DESTINATION, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('dataset_provenance.json', json.dumps(provenance, indent=2))
    with zipfile.ZipFile(DESTINATION) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith('.pt') or name.startswith('data/') for name in archive.namelist())
        count = len(archive.namelist())
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    (ROOT / 'deliverables/submission.sha256').write_text(digest + '  submission.zip\n')
    full_artifacts = sorted(set(files) | set((ROOT / 'results/main').glob('*/best.pt')))
    index = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in full_artifacts]
    (ROOT / 'deliverables/artifact_manifest.json').write_text(json.dumps(index, indent=2))
    print({'files': count, 'bytes': DESTINATION.stat().st_size, 'sha256': digest})


if __name__ == '__main__':
    main()
