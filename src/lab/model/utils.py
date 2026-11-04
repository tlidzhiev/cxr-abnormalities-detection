from torch import nn

_SUPPORTED_ACTIVATIONS = ('relu', 'tanh', 'silu', 'gelu')


def get_activation(activation: str) -> nn.Module:
    """
    Get an activation module by name ('relu', 'tanh', 'silu' or 'gelu').

    Raises
    ------
    ValueError
        If the activation is not supported.
    """
    match activation.lower():
        case 'relu':
            return nn.ReLU()
        case 'tanh':
            return nn.Tanh()
        case 'silu':
            return nn.SiLU()
        case 'gelu':
            return nn.GELU()
        case _:
            supported = ', '.join(f"'{name}'" for name in _SUPPORTED_ACTIVATIONS)
            raise ValueError(
                f"Unknown activation type: '{activation}'. Supported types: {supported}."
            )


def initialize_weights(
    module: nn.Module,
    activation: str,
) -> None:
    """
    Initialize Linear and Conv1d weights: Xavier for tanh, Kaiming otherwise.

    Parameters
    ----------
    module : nn.Module
        Module to initialize.
    activation : str
        Activation name: 'relu', 'tanh', 'silu' or 'gelu'.

    Raises
    ------
    ValueError
        If the activation is not supported.
    """
    activation = activation.lower()
    if activation not in ['relu', 'tanh', 'silu', 'gelu']:
        supported = ', '.join(f"'{name}'" for name in _SUPPORTED_ACTIVATIONS)
        raise ValueError(
            f"Unknown activation type for initialization: '{activation}'. Supported types: {supported}."
        )

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if activation == 'tanh':
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
