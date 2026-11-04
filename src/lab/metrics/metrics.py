"""mAP@IoU (VOC all-point), as in the VinBigData competition (IoU 0.4).

preds: {image_id: {"boxes": [[x1,y1,x2,y2],...] in [0,1], "scores": [...], "labels": [0..K-1]}}
gts:   {image_id: {"boxes": [...], "labels": [...]}}
"""

import numpy as np

NO_FINDING = 14


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def average_precision(recall, precision):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def mean_average_precision(preds, gts, iou_thr=0.4, num_classes=14):
    image_ids = list(gts)
    aps = {}
    for c in range(num_classes):
        gt_c, n_gt = {}, 0
        for i in image_ids:
            b = np.asarray(gts[i]['boxes'], float).reshape(-1, 4)
            m = np.asarray(gts[i]['labels']) == c
            gt_c[i] = b[m] if len(b) else b
            n_gt += int(m.sum())
        if n_gt == 0:
            continue
        dets = []
        for i in image_ids:
            p = preds.get(i)
            if not p or not len(p['boxes']):
                continue
            b = np.asarray(p['boxes'], float).reshape(-1, 4)
            s, l = np.asarray(p['scores'], float), np.asarray(p['labels'])
            for j in np.nonzero(l == c)[0]:
                dets.append((s[j], i, b[j]))
        dets.sort(key=lambda d: -d[0])
        used = {i: np.zeros(len(gt_c[i]), bool) for i in image_ids}
        tp = np.zeros(len(dets))
        for k, (_, i, box) in enumerate(dets):
            g = gt_c[i]
            if len(g) == 0:
                continue
            ious = iou_matrix(box[None], g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not used[i][j]:
                used[i][j] = True
                tp[k] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        recall = ctp / n_gt
        precision = ctp / np.maximum(ctp + cfp, 1e-12)
        aps[c] = average_precision(recall, precision) if len(dets) else 0.0
    return float(np.mean(list(aps.values()))) if aps else 0.0, aps


def gts_from_records(records, min_agree=0.0, with_no_finding=False):
    out = {}
    for r in records:
        fz = r['fused']
        keep = [k for k, s in enumerate(fz['scores']) if s >= min_agree - 1e-6]
        boxes = [fz['boxes'][k] for k in keep]
        labels = [fz['labels'][k] for k in keep]
        if with_no_finding and r['normal']:
            boxes, labels = [[0, 0, 1, 1]], [NO_FINDING]
        out[r['image_id']] = {'boxes': boxes, 'labels': labels}
    return out
