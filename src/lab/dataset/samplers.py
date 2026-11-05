import torch
from torch.utils.data import Dataset, Sampler


class NormalFractionSampler(Sampler[int]):
    """
    Epoch sampler with a fixed share of the images without findings.

    Every pass takes all the images with findings and a new random subset of the images
    without findings (no boxes), so that they make 'normal_frac' of the epoch. The subsets of
    different passes are independent, so over many passes almost all the images without
    findings are used (with 30%: ~68% after 6 passes, >99% after 30). The dataset must
    implement 'is_background()'.

    Each 'iter()' (a new pass of the dataloader) draws a new subset and order.
    """

    def __init__(self, dataset: Dataset, normal_frac: float | None = None, seed: int = 42) -> None:
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset with 'is_background()': a flag per element, True if it has no boxes.
        normal_frac : float or None, optional
            Share of the images without findings in an epoch, in [0, 1). None keeps all of
            them (the natural distribution, only shuffled).
        seed : int, optional
            Seed of the draws (the pass number is added to it).
        """
        background = dataset.is_background()  # ty:ignore[unresolved-attribute]
        self.findings = [i for i, b in enumerate(background) if not b]
        self.normal = [i for i, b in enumerate(background) if b]
        if normal_frac is None:
            self.num_normal = len(self.normal)
        else:
            assert 0 <= normal_frac < 1, 'normal_frac must be in [0, 1).'
            wanted = round(len(self.findings) * normal_frac / (1 - normal_frac))
            self.num_normal = min(len(self.normal), wanted)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.findings) + self.num_normal

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        normal = torch.tensor(self.normal, dtype=torch.long)
        normal = normal[torch.randperm(len(normal), generator=generator)[: self.num_normal]]
        indices = torch.cat([torch.tensor(self.findings, dtype=torch.long), normal])
        indices = indices[torch.randperm(len(indices), generator=generator)]
        return iter(indices.tolist())
