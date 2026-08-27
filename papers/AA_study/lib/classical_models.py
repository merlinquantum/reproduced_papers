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
        Simple CNN for square images.

        Parameters
        ----------
        input_channels : int, optional
            Number of input channels.
        input_image_size : int, optional
            Spatial size (width) of square inputs.
        num_classes : int, optional
            Number of output classes.
        kernel_size : int, optional
            Convolution and pooling kernel size.
        stride : int, optional
            Stride for the first max-pooling layer.
        num_layers : int, optional
            Number of convolutional blocks.
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

        feature_extractor = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_image_size, input_image_size)
            flat_dim = feature_extractor(dummy).view(1, -1).shape[1]

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(
                    flat_dim,
                    2 * num_classes,
                ),
                nn.ReLU(),
                nn.Linear(2 * num_classes, num_classes),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (N, num_classes).
        """
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)
        return self.model(x)
