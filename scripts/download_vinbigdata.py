"""Download the VinBigData chest X-ray dataset from Kaggle.

Files are placed into ``data/vinbigdata/raw/{image-size}``.

Usage:
    uv run python scripts/download_vinbigdata.py --image-size 512
"""

import argparse
from pathlib import Path

import kagglehub  # ty:ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'vinbigdata' / 'raw'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--image-size',
        type=int,
        default=512,
        help='Resized image side in pixels, selects the Kaggle dataset (default: 512).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DATA_DIR,
        help=f'Root directory for raw data (default: {DATA_DIR}).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if the target directory already exists.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handle = f'awsaf49/vinbigdata-{args.image_size}-image-dataset'
    target = args.output_dir / str(args.image_size)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {handle} -> {target}')
    path = kagglehub.dataset_download(handle, output_dir=str(target), force_download=args.force)
    print('Path to dataset files:', path)


if __name__ == '__main__':
    main()
