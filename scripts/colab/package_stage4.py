"""Build the portable Colab source bundle; never include imagery or local secrets."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    paths = [ROOT / 'geocadastra/__init__.py']
    for folder in ('geocadastra/models', 'geocadastra/synth'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    paths.extend(ROOT / name for name in (
        'scripts/colab/train_stage4_gpu.py', 'scripts/colab/requirements.txt',
        'notebooks/GeoCadastra_Stage4_GPU.ipynb', 'notebooks/STAGE4_COLAB_README.md'))
    out = ROOT / 'artifacts/stage4_gpu_bundle.zip'
    out.parent.mkdir(exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr('source_checksums.json', json.dumps(hashes, indent=2))
    print(out)


if __name__ == '__main__':
    main()
