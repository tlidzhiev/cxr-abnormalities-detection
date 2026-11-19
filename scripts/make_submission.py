"""Build a Kaggle submission (VinBigData) from the test predictions of one or more models.

Input: 'predictions.json' of the 'test' partition made by lab.inference, one per model (e.g. one
per fold): {image_id: {boxes [N,4] normalized xyxy, scores [N], labels [N] in 0..13}}.
The models are merged with Weighted Boxes Fusion, the boxes are scaled to the pixels of the
original images (width/height from test.csv), and one row per image of sample_submission.csv is
written: 'PredictionString' = 'class_id score x_min y_min x_max y_max ...'. Class 14 is
"No finding": an image without boxes gets '14 1 0 0 1 1', as in the sample submission.

Example (5 folds, see README):
    uv run python scripts/make_submission.py \
        --predictions data/saved/frcnn_fold{0,1,2,3,4}/test/predictions.json \
        --output submission.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from lab.utils.io import read_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / 'data' / 'vinbigdata' / 'raw' / '512' / 'vinbigdata'
NO_FINDING = 14
NO_FINDING_STRING = f'{NO_FINDING} 1 0 0 1 1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--predictions',
        nargs='+',
        type=Path,
        required=True,
        help='predictions.json of the test partition of each model (or its run directory).',
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Weight of each model in the fusion (default: equal).',
    )
    parser.add_argument('--output', type=Path, default=ROOT / 'submission.csv')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Directory with test.csv (original image sizes) and sample_submission.csv.',
    )
    parser.add_argument('--iou-thr', type=float, default=0.4, help='WBF IoU threshold.')
    parser.add_argument(
        '--skip-box-thr',
        type=float,
        default=0.0,
        help='WBF: drop the boxes of the models with a lower score before the fusion.',
    )
    parser.add_argument(
        '--min-score', type=float, default=0.001, help='Drop the fused boxes below this score.'
    )
    parser.add_argument('--max-boxes', type=int, default=100, help='Max boxes per image.')
    parser.add_argument(
        '--add-no-finding',
        action='store_true',
        help="Also add '14 (1 - max score) 0 0 1 1' to the images with boxes (a heuristic "
        "confidence of 'No finding'; there is no classifier for it).",
    )
    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help="Fill the images absent from the predictions with '14 1 0 0 1 1' instead of "
        'failing (for smoke tests on a subset only).',
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, list]]:
    if path.is_dir():
        path = path / 'test' / 'predictions.json'
    return read_json(path)  # ty:ignore[invalid-explicit-return]


def fuse(
    preds: list[dict[str, list]],
    weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge the predictions of the models for one image: normalized boxes, scores, labels.
    """
    boxes = [np.asarray(p['boxes'], dtype=np.float64).reshape(-1, 4) for p in preds]
    scores = [np.asarray(p['scores'], dtype=np.float64) for p in preds]
    labels = [np.asarray(p['labels'], dtype=np.float64) for p in preds]
    if all(len(b) == 0 for b in boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    # conf_type='avg' divides the score by the number of models, so that a box found by
    # one model only gets a lower score than a box that the models agree on
    return weighted_boxes_fusion(
        boxes,
        scores,
        labels,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type='avg',
    )


def prediction_string(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
    min_score: float,
    max_boxes: int,
    add_no_finding: bool,
) -> str:
    keep = scores >= min_score
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    order = np.argsort(-scores)[:max_boxes]
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    if len(boxes) == 0:
        return NO_FINDING_STRING

    scale = np.array([width, height, width, height])
    parts = []
    for box, score, label in zip(boxes * scale, scores, labels, strict=True):
        x1, y1, x2, y2 = box
        parts.append(f'{int(label)} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}')
    if add_no_finding:
        parts.append(f'{NO_FINDING} {1 - scores.max():.4f} 0 0 1 1')
    return ' '.join(parts)


def main() -> None:
    args = parse_args()
    weights = args.weights or [1.0] * len(args.predictions)
    if len(weights) != len(args.predictions):
        raise SystemExit('--weights must have one value per --predictions.')

    models = [load_predictions(path) for path in args.predictions]
    with (args.data_dir / 'sample_submission.csv').open() as handle:
        image_ids = [row['image_id'] for row in csv.DictReader(handle)]
    with (args.data_dir / 'test.csv').open() as handle:
        sizes = {r['image_id']: (int(r['width']), int(r['height'])) for r in csv.DictReader(handle)}

    for path, model in zip(args.predictions, models, strict=True):
        missing = [i for i in image_ids if i not in model]
        if missing and not args.allow_missing:
            raise SystemExit(
                f'{path}: no predictions for {len(missing)} of {len(image_ids)} test images '
                f'(e.g. {missing[0]}). Run lab.inference on the whole test partition, or pass '
                '--allow-missing for a smoke test.'
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    num_empty = 0
    with args.output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_id', 'PredictionString'])
        for image_id in image_ids:
            available = [
                (m[image_id], w) for m, w in zip(models, weights, strict=True) if image_id in m
            ]
            if not available:
                writer.writerow([image_id, NO_FINDING_STRING])
                num_empty += 1
                continue
            preds, used_weights = zip(*available, strict=True)
            boxes, scores, labels = fuse(
                list(preds), list(used_weights), args.iou_thr, args.skip_box_thr
            )
            width, height = sizes[image_id]
            text = prediction_string(
                boxes,
                scores,
                labels,
                width,
                height,
                args.min_score,
                args.max_boxes,
                args.add_no_finding,
            )
            writer.writerow([image_id, text])
            num_empty += text == NO_FINDING_STRING

    print(
        f'Wrote {args.output}: {len(image_ids)} images, {len(models)} model(s), '
        f'{num_empty} without boxes ("{NO_FINDING_STRING}").'
    )


if __name__ == '__main__':
    main()
