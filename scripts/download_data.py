"""Download the original MovieLens 1M archive and verify GroupLens' checksum."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
EXPECTED_MD5 = 'c4d9eecfca2ab87c1945afe126590906'


def main():
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'ml-1m.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError(f'MovieLens archive checksum mismatch: {digest}')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (raw / member.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(raw)
    print(f'PASS official MovieLens 1M MD5={digest}')


if __name__ == '__main__':
    main()
