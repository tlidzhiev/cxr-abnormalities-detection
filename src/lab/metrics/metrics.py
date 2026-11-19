"""mAP@IoU (VOC all-point), as in the VinBigData competition (IoU 0.4).

preds: {image_id: {"boxes": [[x1,y1,x2,y2],...] in [0,1], "scores": [...], "labels": [0..K-1]}}
gts:   {image_id: {"boxes": [...], "labels": [...]}}

The values may be lists, numpy arrays or tensors. Everything is computed in torch on the
device of the input tensors (CPU otherwise), vectorized over all the images and classes, with
two device-to-host syncs per call.
"""

import torch

NO_FINDING = 14


def iou_matrix(a, b) -> torch.Tensor:
    """
    IoU of each box of 'a' (N, 4) with each box of 'b' (M, 4), xyxy -> (N, M).
    """
    a, b = torch.as_tensor(a).reshape(-1, 4), torch.as_tensor(b).reshape(-1, 4)
    return _iou(a[:, None], b[None, :])


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU of the broadcast boxes 'a' (..., 4) and 'b' (..., 4).
    """
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-12)


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    """
    VOC all-point AP: the area under the precision envelope (the max precision at any
    higher recall), as a 0-dim tensor (no host sync).
    """
    envelope = precision.flip(0).cummax(0).values.flip(0)
    # The recall steps; the steps after the last detection have zero precision
    steps = torch.diff(recall, prepend=recall.new_zeros(1))
    return (steps * envelope).sum()


def _device(*collections: dict) -> torch.device:
    """
    Device of the first tensor in the preds / gts (CPU if there is none).
    """
    for collection in collections:
        for item in collection.values():
            for value in (item or {}).values():
                if isinstance(value, torch.Tensor):
                    return value.device
    return torch.device('cpu')


def mean_average_precision(preds, gts, iou_thr=0.4, num_classes=14):
    """
    mAP over the classes with a ground truth (and AP of each of them as {class: AP}).

    The matching is the greedy VOC one: the detections of a class are taken by descending
    score, each is matched to the ground-truth box of its image with the highest IoU, and it
    is a TP if that IoU >= 'iou_thr' and the box is not matched yet (otherwise a FP).
    This is vectorized: a detection is a TP iff its best IoU >= 'iou_thr' and it is the first
    such detection (by descending score) with that best box, since exactly that detection
    marks the box as matched.

    Labels outside [0, num_classes) are ignored.
    """
    image_ids = list(gts)
    if not image_ids:
        return 0.0, {}
    device = _device(preds, gts)
    # float64 as numpy (MPS has no float64)
    dtype = torch.float32 if device.type == 'mps' else torch.float64
    K = num_classes

    def as_boxes(x):
        return torch.as_tensor(x, dtype=dtype, device=device).reshape(-1, 4)

    def as_vector(x, dt):
        return torch.as_tensor(x, dtype=dt, device=device).reshape(-1)

    def image_index(counts: list[int]) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            torch.tensor(counts, device=device),
            output_size=sum(counts),
        )

    def count_by(index: torch.Tensor, size: int) -> torch.Tensor:
        # 'bincount' with the known size ('torch.bincount' syncs on CUDA to find it)
        return torch.zeros(size, dtype=torch.long, device=device).scatter_add_(
            0, index, torch.ones_like(index)
        )

    def class_index(labels: torch.Tensor) -> torch.Tensor:
        # Labels outside the classes go to the ignored class K (without boolean indexing,
        # which syncs)
        return torch.where((labels >= 0) & (labels < K), labels, K)

    # Ground truth of all the images, concatenated
    gt_boxes = [as_boxes(gts[i]['boxes']) for i in image_ids]
    gt_counts = [len(b) for b in gt_boxes]
    gt_box = torch.cat(gt_boxes)
    gt_cls = class_index(torch.cat([as_vector(gts[i]['labels'], torch.long) for i in image_ids]))
    gt_key = image_index(gt_counts) * (K + 1) + gt_cls

    # Detections of all the images, concatenated (in the order of the images)
    det = [preds.get(i) or {} for i in image_ids]
    det_boxes = [as_boxes(p.get('boxes', [])) for p in det]
    det_counts = [len(b) for b in det_boxes]
    det_box = torch.cat(det_boxes)
    det_score = torch.cat(
        [as_vector(p['scores'], dtype) for p, n in zip(det, det_counts, strict=True) if n]
        or [det_box.new_zeros(0)]
    )
    det_cls = class_index(
        torch.cat(
            [as_vector(p['labels'], torch.long) for p, n in zip(det, det_counts, strict=True) if n]
            or [gt_cls.new_zeros(0)]
        )
    )
    det_key = image_index(det_counts) * (K + 1) + det_cls

    # Sync 1: the numbers of detections and GT boxes of each class, and the max number of
    # GT boxes of an (image, class)
    sizes = torch.cat(
        [
            count_by(det_cls, K + 1),
            count_by(gt_cls, K + 1),
            count_by(gt_key, len(image_ids) * (K + 1)).max().reshape(1),
        ]
    ).tolist()
    n_det, n_gt, max_group = sizes[: K + 1], sizes[K + 1 : 2 * K + 2], sizes[-1]
    if not any(n_gt[:K]):
        return 0.0, {}

    # The GT boxes of the (image, class) of each detection: a range of the GT boxes sorted
    # by (image, class), keeping their order in the image (for the argmax ties)
    gt_order = torch.argsort(gt_key, stable=True)
    gt_key, gt_box = gt_key[gt_order], gt_box[gt_order]
    start = torch.searchsorted(gt_key, det_key)
    end = torch.searchsorted(gt_key, det_key, right=True)
    candidates = start[:, None] + torch.arange(max_group, device=device)
    is_candidate = candidates < end[:, None]
    candidates = candidates.clamp(max=len(gt_key) - 1)
    ious = _iou(det_box[:, None], gt_box[candidates])
    ious = torch.where(is_candidate, ious, -torch.inf)
    best = ious.argmax(1, keepdim=True)
    best_iou = ious.gather(1, best).squeeze(1)
    best_gt = candidates.gather(1, best).squeeze(1)
    matched = best_iou >= iou_thr

    # Detections sorted by class, then by descending score (ties in the image order)
    order = torch.argsort(det_score, descending=True, stable=True)
    order = order[torch.argsort(det_cls[order], stable=True)]
    matched, best_gt = matched[order], best_gt[order]

    # A matched detection is a TP iff it is the first one matched to its GT box
    position = torch.arange(len(order), device=device)
    first = torch.full((len(gt_key),), len(order), device=device).scatter_reduce(
        0, best_gt, torch.where(matched, position, len(order)), reduce='amin'
    )
    tp = (matched & (first[best_gt] == position)).to(dtype)

    # AP of each class with a ground truth, from its slice of the sorted detections
    classes, aps = [], []
    offset = 0
    for c in range(K):
        n = n_det[c]
        if n_gt[c]:
            classes.append(c)
            if n == 0:
                aps.append(tp.new_zeros(()))
            else:
                ctp = tp[offset : offset + n].cumsum(0)
                rank = torch.arange(1, n + 1, dtype=dtype, device=device)
                aps.append(average_precision(ctp / n_gt[c], ctp / rank))
        offset += n

    # Sync 2: the APs
    ap = torch.stack(aps)
    values = torch.cat([ap.mean().reshape(1), ap]).tolist()
    return values[0], dict(zip(classes, values[1:], strict=True))


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
