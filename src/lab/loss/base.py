from torch import nn


class BaseLoss(nn.Module):
    """
    Base class for all loss functions.

    Attributes
    ----------
    loss_names : list[str]
        Names of all loss components.
    """

    loss_names: list[str]
