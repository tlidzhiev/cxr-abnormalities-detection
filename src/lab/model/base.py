from torch import nn


class BaseModel(nn.Module):
    """
    Base class for all models.
    """

    def __str__(self) -> str:
        """
        Model prints with the number of parameters.
        """

        def format_number(num: int) -> str:
            if abs(num) >= 1_000_000:
                return f'{num / 1_000_000:.2f}M'
            elif abs(num) >= 1_000:
                return f'{num / 1_000:.2f}K'
            else:
                return str(num)

        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info += f'\nAll parameters: {format_number(all_parameters)}'
        result_info += f'\nTrainable parameters: {format_number(trainable_parameters)}'
        return result_info
