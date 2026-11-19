import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image


def caption_image(
    img: torch.Tensor,
    text: str,
    color: str = 'black',
    scale: int = 4,
    bar_height: int = 14,
) -> torch.Tensor:
    """Upscale an image and draw a caption above it."""
    img = img.detach().float().cpu()
    img = (img - img.min()) / (img.max() - img.min()).clamp_min(1e-8)
    img = F.interpolate(img[None], scale_factor=scale, mode='nearest')[0]
    pil_img = to_pil_image(img).convert('RGB')

    canvas = Image.new('RGB', (pil_img.width, pil_img.height + bar_height), 'white')
    canvas.paste(pil_img, (0, bar_height))
    ImageDraw.Draw(canvas).text((2, 1), text, fill=color)
    return pil_to_tensor(canvas)
