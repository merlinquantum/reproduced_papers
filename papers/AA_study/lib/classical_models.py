import torch.nn as nn


class CNN(nn.module):
    def __init__(
        self,
        input_channels: int = 1,
        input_image_size: int = 32,
        kernel_size: int = 5,
        stride: int = 2,
        num_layers: int = 2,
    ):
        """
        input_image_size: only one side, images must be squares
        """
        super().__init__()
        layers = []
        layers.extend(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_image_size,
                kernel_size=kernel_size,
            )
        )
        layers.extend(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        for i in range(0, num_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        input_image_size * (2 ** (i)),
                        input_image_size * (2 ** (i + 1)),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear((input_image_size**3) // (2 ** (num_layers + 1)), 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            ]
        )
        self.model = nn.Sequential(*layers)
