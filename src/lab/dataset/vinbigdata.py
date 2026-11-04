import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from lab.utils.io import get_root, read_json

from .base import BaseDataset

logger = logging.getLogger(__name__)


class VinBigDataDataset(BaseDataset):
    """
    VinBigData chest X-ray detection dataset.

    Reads 'annotations.json' made by 'scripts/preprocess_vinbigdata.py' and the PNG images
    of 'data/vinbigdata/raw/{size}/vinbigdata/train'. The 'val' split is the held-out fold,
    the 'train' split is all the other folds.

    An instance is a dict:
        img      : float32 tensor (3, H, W) in [0, 1];
        boxes    : float32 tensor (N, 4), xyxy in pixels of the loaded image
                   (N = 0 for normal images);
        labels   : int64 tensor (N,), class ids 0..13 (no background class);
        image_id : str.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'val'] = 'train',
        fold: int = 0,
        size: int = 512,
        annotations: Path | str | None = None,
        min_agree: float = 0.0,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str or None, optional
            Dataset directory relative to the project root. If None, 'data/vinbigdata'.
        split : {'train', 'val'}, optional
            'val' is the fold 'fold', 'train' is all the other folds.
        fold : int, optional
            Held-out fold.
        size : int, optional
            Which downloaded dataset size to read images from (see download_vinbigdata.py).
        annotations : Path or str or None, optional
            Annotations file relative to the project root. If None,
            '{root}/preprocessed/annotations.json'.
        min_agree : float, optional
            Drop the boxes that a smaller fraction of radiologists agreed on (0.6 keeps the
            boxes marked by at least 2 of 3). Use 0 for validation.
        limit : int or None, optional
            Limit the number of samples.
        shuffle_index : bool, optional
            Whether to shuffle the index.
        instance_transforms : Callable or None, optional
            Transform of the whole instance with numpy fields (see 'load_instance'), e.g.
            'DetectionTransform' with an Albumentations pipeline (image and boxes together).
        """
        root = get_root() / (root if root is not None else 'data/vinbigdata')
        ann_path = (
            get_root() / annotations if annotations else root / 'preprocessed' / 'annotations.json'
        )
        img_dir = root / 'raw' / str(size) / 'vinbigdata' / 'train'
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f'No images in {img_dir}. Run scripts/download_vinbigdata.py --image-size {size}.'
            )

        content = read_json(ann_path)
        self.classes: list[str] = content['classes']  # ty:ignore[invalid-argument-type]
        records: list[dict[str, Any]] = content['records']  # ty:ignore[invalid-argument-type]

        if split == 'val':
            records = [r for r in records if r['fold'] == fold]
        elif split == 'train':
            records = [r for r in records if r['fold'] != fold]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}.")

        index = []
        for rec in records:
            fz = rec['fused']
            # Keep boxes with enough agreement, and drop degenerate ones (< 1 px at this
            # size: they would break the detector losses). Images are resized to size x size.
            keep = [
                k
                for k, (box, score) in enumerate(zip(fz['boxes'], fz['scores'], strict=True))
                if score >= min_agree - 1e-6
                and (box[2] - box[0]) * size >= 1
                and (box[3] - box[1]) * size >= 1
            ]
            index.append(
                {
                    'path': str(img_dir / f'{rec["image_id"]}.png'),
                    'image_id': rec['image_id'],
                    'boxes': [fz['boxes'][k] for k in keep],  # normalized xyxy
                    'labels': [fz['labels'][k] for k in keep],
                    'normal': rec['normal'],
                }
            )
        logger.info(f'VinBigData {split} (fold {fold}): {len(index)} images')

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def load_instance(self, entry: dict[str, Any]) -> dict[str, Any]:
        """
        Load the image and scale the normalized boxes to its pixels.

        Returns numpy fields (for the instance transforms): 'img' (H, W, 3) uint8,
        'boxes' (N, 4) float32 xyxy in pixels, 'labels' (N,) int64.
        """
        img = np.array(Image.open(entry['path']).convert('RGB'))
        h, w = img.shape[:2]
        boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 4)
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)
        labels = np.asarray(entry['labels'], dtype=np.int64)
        return {'img': img, 'boxes': boxes, 'labels': labels, 'image_id': entry['image_id']}

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the instance transform, then convert the numpy fields to tensors:
        'img' to float32 (3, H, W) in [0, 1].
        """
        instance_data = super().preprocess_data(instance_data)
        return {
            'img': torch.from_numpy(instance_data['img']).permute(2, 0, 1).float() / 255,
            'boxes': torch.from_numpy(instance_data['boxes']),
            'labels': torch.from_numpy(instance_data['labels']),
            'image_id': instance_data['image_id'],
        }
