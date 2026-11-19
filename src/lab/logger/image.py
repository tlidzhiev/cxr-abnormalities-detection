import torch
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes

# One color per class
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000',
]  # fmt: skip


def draw_detections(
    img: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None = None,
    classes: list[str] | None = None,
    tile_size: int = 512,
) -> torch.Tensor:
    """
    Draw the boxes on an image (3, H, W) in [0, 1]; returns a uint8 tile (3, S, S).
    """
    tile = (img.detach().float().cpu() * 255).round().to(torch.uint8)
    if len(boxes) > 0:  # normal images have no boxes
        texts = [
            (classes[c] if classes else str(c)) + ('' if scores is None else f' {s:.2f}')
            for c, s in zip(
                labels.tolist(),
                scores.tolist() if scores is not None else [0.0] * len(labels),
                strict=True,
            )
        ]
        tile = draw_bounding_boxes(
            tile,
            boxes.detach().float().cpu(),
            texts,
            colors=[COLORS[c % len(COLORS)] for c in labels.tolist()],
            width=3,
        )
    return F.interpolate(tile[None].float(), size=(tile_size, tile_size))[0].byte()
