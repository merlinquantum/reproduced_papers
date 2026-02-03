import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_image_size: int = 32,
        num_classes: int = 2,
        kernel_size: int = 5,
        stride: int = 2,
        num_layers: int = 2,
    ):
        """
        input_image_size: only one side, images must be squares
        """
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_image_size,
                kernel_size=kernel_size,
            )
        )
        layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        for i in range(0, num_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        input_image_size * (2 ** (i)),
                        input_image_size * (2 ** (i + 1)),
                        kernel_size=kernel_size,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(
                    (input_image_size ** (2 * input_channels))
                    // (2 ** (num_layers + 1)),
                    2 * num_classes,
                ),
                nn.ReLU(),
                nn.Linear(2 * num_classes, num_classes),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)
        return self.model(x)
