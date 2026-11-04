"""Annotation preparation: merge radiologist boxes (WBF) + multilabel stratified k-fold.

Input: train.csv from the VinBigData dataset (awsaf49):
    image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max, width, height
    (class_id 14 = "No finding", such rows have no boxes; width/height is the original image
    size, the boxes are given in the same pixels). We assume image_id == patient, so folds are
    stratified per image.

Output (in data/vinbigdata/preprocessed/): params.json (hyperparameters), summary.txt (fold
distribution) and annotations.json, a list of per-image records:
    image_id, width, height, fold, group, normal,
    rads   - which radiologists annotated the image,
    soft   - 15 numbers: fraction of radiologists who marked each of the 14 classes, plus the
             fraction who marked anything at all,
    fused  - WBF-merged boxes {boxes [N,4] in [0,1], labels 0..13, scores = fraction of agreeing
             radiologists},
    raw    - original boxes of each radiologist (needed for "annotator sampling").

uv run python scripts/preprocess_vinbigdata.py [--size 512] [--folds 5] [--wbf-iou 0.4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 14
NO_FINDING = 14
CLASS_NAMES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
]


def fuse_image(df_img, w, h, iou_thr):
    """WBF across radiologists within each class. The fused box score is the fraction of
    radiologists who marked that box (conf_type='avg' divides by the number of "models" =
    radiologists)."""
    rads = sorted(df_img.rad_id.unique())
    raw, boxes_list, scores_list, labels_list = {}, [], [], []
    for r in rads:
        d = df_img[(df_img.rad_id == r) & (df_img.class_id != NO_FINDING)]
        b = d[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy(np.float64)
        b = b / np.array([w, h, w, h]) if len(b) else np.zeros((0, 4))
        b = np.clip(b, 0, 1)
        lab = d.class_id.to_numpy(int)
        raw[r] = {'boxes': b.round(5).tolist(), 'labels': lab.tolist()}
        boxes_list.append(b.tolist())
        scores_list.append([1.0] * len(b))
        labels_list.append(lab.tolist())

    if sum(len(b) for b in boxes_list) == 0:
        fused = {'boxes': [], 'labels': [], 'scores': []}
    else:
        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=0.0, conf_type='avg'
        )
        fused = {
            'boxes': np.round(fb, 5).tolist(),
            'labels': fl.astype(int).tolist(),
            'scores': np.round(fs, 4).tolist(),
        }

    soft = np.zeros(NUM_CLASSES + 1)
    for r in rads:
        labs = set(raw[r]['labels'])
        for c in labs:
            soft[c] += 1
        soft[NUM_CLASSES] += len(labs) > 0
    soft /= len(rads)
    return rads, raw, fused, soft


def make_folds(records, n_folds, seed):
    """Multilabel stratified k-fold over groups (patient / duplicate cluster)."""
    all_rads = sorted({r for rec in records for r in rec['rads']})
    rad_idx = {r: i for i, r in enumerate(all_rads)}
    feats = []
    for rec in records:
        v = np.zeros(NUM_CLASSES + 1 + 4 + len(all_rads))
        v[:NUM_CLASSES] = (
            np.array(rec['soft'][:NUM_CLASSES]) > 0
        )  # classes (marked by at least one radiologist)
        v[NUM_CLASSES] = rec['normal']  # normal
        n = len(rec['fused']['boxes'])  # bins of the box count
        v[NUM_CLASSES + 1 + (0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3)] = 1
        for r in rec['rads']:  # who annotated
            v[NUM_CLASSES + 5 + rad_idx[r]] = 1
        feats.append(v)
    feats = np.array(feats)

    groups = np.array([rec['group'] for rec in records])
    uniq, inv = np.unique(groups, return_inverse=True)
    g_feats = np.zeros((len(uniq), feats.shape[1]))
    np.maximum.at(g_feats, inv, feats)  # aggregate to groups

    g_fold = np.zeros(len(uniq), int)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(mskf.split(np.zeros(len(uniq)), g_feats)):
        g_fold[va] = f
    for rec, gi in zip(records, inv):
        rec['fold'] = int(g_fold[gi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--size',
        type=int,
        default=512,
        help='downloaded dataset size to read train.csv from (output does not depend on it)',
    )
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument(
        '--out-dir', type=Path, default=None, help='default: data/vinbigdata/preprocessed'
    )
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--wbf-iou', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    raw_dir = ROOT / 'data' / 'vinbigdata' / 'raw' / str(args.size) / 'vinbigdata'
    csv_path = args.csv or raw_dir / 'train.csv'
    out_dir = args.out_dir or ROOT / 'data' / 'vinbigdata' / 'preprocessed'

    df = pd.read_csv(csv_path)
    sizes = df.groupby('image_id')[['width', 'height']].nunique()
    assert (sizes == 1).all().all(), 'an image has several different width/height values'  # ty:ignore[unresolved-attribute]
    sizes = df.groupby('image_id')[['width', 'height']].first()

    images_dir = raw_dir / 'train'
    if images_dir.is_dir():
        on_disk = {p.stem for p in images_dir.glob('*.png')}
        missing = set(sizes.index) - on_disk
        assert not missing, f'{len(missing)} images from train.csv have no file on disk'

    records = []
    for image_id, d in df.groupby('image_id', sort=True):
        w, h = int(sizes.at[image_id, 'width']), int(sizes.at[image_id, 'height'])
        rads, raw, fused, soft = fuse_image(d, w, h, args.wbf_iou)
        records.append(
            {
                'image_id': image_id,
                'width': w,
                'height': h,
                'group': image_id,  # image_id == patient
                'normal': bool(soft[NUM_CLASSES] == 0),
                'rads': rads,
                'soft': np.round(soft, 4).tolist(),
                'fused': fused,
                'raw': raw,
            }
        )
    make_folds(records, args.folds, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'annotations.json'
    with open(out_path, 'w') as f:
        json.dump({'classes': CLASS_NAMES, 'records': records}, f)

    # short summary: distribution over folds
    tab = pd.DataFrame(
        [
            {
                'fold': r['fold'],
                'normal': r['normal'],
                **{c: int(c_i in r['fused']['labels']) for c_i, c in enumerate(CLASS_NAMES)},  # ty:ignore[not-subscriptable]
            }
            for r in records
        ]
    )
    summary = f'{len(records)} images, normal: {tab.normal.mean():.1%}\n'
    summary += tab.groupby('fold').sum().T.to_string()
    print(summary)
    (out_dir / 'summary.txt').write_text(summary + '\n')

    params = {
        'size': args.size,
        'csv': str(csv_path),
        'folds': args.folds,
        'wbf_iou': args.wbf_iou,
        'seed': args.seed,
        'num_images': len(records),
    }
    (out_dir / 'params.json').write_text(json.dumps(params, indent=2) + '\n')
    print(f'Saved annotations.json, summary.txt, params.json to {out_dir}')


if __name__ == '__main__':
    main()
