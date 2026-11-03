from collections.abc import Callable

import torch
from torch import Tensor, nn


class MappingNetwork(nn.Module):
    """Mapping network to transform latent vectors.

    Args:
        z_dim: Dimensionality of the input latent vector.
        hidden_dim: Dimensionality of the hidden layers.
        num_layers: Number of layers in the mapping network.
        use_rms_norm: Whether to apply RMS normalization to the input.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        self.rms_norm = nn.RMSNorm([z_dim], eps=1e-6)
        self.use_rms_norm = use_rms_norm

        start = torch.log2(torch.tensor(z_dim))
        stop = torch.log2(torch.tensor(hidden_dim))

        hidden_dims = torch.logspace(
            start, stop, num_layers + 1, base=2, dtype=torch.int
        ).tolist()[1:]

        layers = nn.ModuleList()

        input_dim = z_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        if self.use_rms_norm:
            z = self.rms_norm(z)
        return self.network(z)


class G_block(nn.Module):
    """A single block for the generator network.

    Args:
        out_channels: Number of output channels.
        in_channels: Number of input channels.
        kernel_size: Size of the convolutional kernel.
        strides: Stride of the convolution.
        padding: Padding for the convolution.
        norm_layer: Normalization layer to use.
        activation_layer: Activation function to use.
    """

    def __init__(
        self,
        out_channels,
        in_channels=1,
        kernel_size=4,
        strides=2,
        padding=1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.SiLU,
    ):
        super().__init__()
        block = nn.ModuleList()

        block.append(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, strides, padding, bias=False
            )
        )

        if norm_layer is not None:
            block.append(norm_layer(out_channels))

        if activation_layer is not None:
            block.append(activation_layer())

        self.net = nn.Sequential(*block)

    def forward(self, X):
        return self.net(X)


class D_block(nn.Module):
    """A single block for the discriminator network.

    Args:
        out_channels: Number of output channels.
        in_channels: Number of input channels.
        kernel_size: Size of the convolutional kernel.
        strides: Stride of the convolution.
        padding: Padding for the convolution.
        norm_layer: Normalization layer to use.
        activation_layer: Activation function to use.
    """

    def __init__(
        self,
        out_channels,
        in_channels=1,
        kernel_size=4,
        strides=2,
        padding=1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.SiLU,
    ):
        super().__init__()
        block = nn.ModuleList()
        block.append(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, strides, padding, bias=False
            )
        )
        if norm_layer is not None:
            block.append(norm_layer(out_channels))

        if activation_layer is not None:
            block.append(activation_layer())
        self.net = nn.Sequential(*block)

    def forward(self, X):
        return self.net(X)


class ConvGenerator(nn.Module):
    """Convolutional Generator Network.

    Args:
        hidden_dim: Dimensionality of the hidden layers.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        start = torch.log2(torch.tensor(hidden_dim))
        stop = torch.log2(torch.tensor(9))
        gb_hidden_channels_tensor = (
            torch.logspace(
                start,
                stop,
                6,
                base=2,
            )
            .ceil()
            .to(torch.int)
        )

        gb_hidden_channels = gb_hidden_channels_tensor.tolist()

        self.main = nn.ModuleList()
        self.main.append(
            G_block(
                in_channels=gb_hidden_channels[0],
                out_channels=gb_hidden_channels[1],
                kernel_size=2,
                strides=2,
                padding=0,
            )
        )  # Output: (gb_hidden_channels[1], 2))
        self.main.append(
            G_block(
                in_channels=gb_hidden_channels[1],
                out_channels=gb_hidden_channels[2],
                kernel_size=2,
                strides=2,
                padding=0,
            )
        )  # Output: (gb_hidden_channels[2], 4)
        self.main.append(
            G_block(
                in_channels=gb_hidden_channels[2],
                out_channels=gb_hidden_channels[3],
                kernel_size=2,
                strides=2,
                padding=0,
            )
        )  # Output: (gb_hidden_channels[3], 8)
        self.main.append(
            G_block(
                in_channels=gb_hidden_channels[3],
                out_channels=gb_hidden_channels[4],
                kernel_size=2,
                strides=2,
                padding=0,
            )
        )  # Output: (gb_hidden_channels[4], 16)
        self.main.append(
            nn.ConvTranspose1d(
                in_channels=gb_hidden_channels[4],
                out_channels=gb_hidden_channels[5],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )
        )  # Output: (gb_hidden_channels[5], 32)
        self.main.append(nn.Linear(32, 22))  # Output: (gb_hidden_channels[5], 22)

        self.maps = nn.ModuleList()
        linear_sizes = (
            gb_hidden_channels_tensor
            * torch.logspace(0.0, 5.0, 6, base=2, dtype=torch.int)
        ).tolist()
        idx = 1
        for layer in self.main:
            if isinstance(layer, nn.ConvTranspose1d):
                self.maps.append(nn.Linear(hidden_dim, linear_sizes[idx]))
            elif isinstance(layer, G_block):
                # G_block wraps its layers in self.net (an nn.Sequential)
                for sublayer in layer.net:
                    if isinstance(sublayer, nn.ConvTranspose1d):
                        self.maps.append(nn.Linear(hidden_dim, linear_sizes[idx]))
            idx += 1

    def forward(self, h, z_feat):
        input = h.unsqueeze(-1)
        input = self.main[0](input)
        for layer, conv in zip(self.main[1:-1], self.maps[:-1]):
            noise = conv(z_feat).view(input.shape)
            input = input + noise
            input = layer(input)
        output = self.main[-1](input)

        return output


class ConvDiscriminator(nn.Module):
    """Convolutional Discriminator Network.

    Args:
        hidden_dim: Dimensionality of the hidden layers.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # start = torch.log2(torch.tensor(32))
        # stop = torch.log2(torch.tensor(hidden_dim))
        # db_hidden_channels_tensor = (
        #     torch.logspace(
        #         start,
        #         stop,
        #         5,
        #         base=2,
        #     )
        #     .ceil()
        #     .to(torch.int)
        # )

        # db_hidden_channels = db_hidden_channels_tensor.tolist()
        db_hidden_channels = [32, 64, 128, 64, 32]

        self.main = nn.ModuleList()
        self.main.append(
            D_block(
                in_channels=1,
                out_channels=db_hidden_channels[0],
                kernel_size=4,
                strides=2,
                padding=1,
            )
        )
        self.main.append(
            D_block(
                in_channels=db_hidden_channels[0],
                out_channels=db_hidden_channels[1],
                kernel_size=4,
                strides=2,
                padding=1,
            )
        )
        self.main.append(
            D_block(
                in_channels=db_hidden_channels[1],
                out_channels=db_hidden_channels[2],
                kernel_size=4,
                strides=2,
                padding=1,
            )
        )
        self.main.append(
            D_block(
                in_channels=db_hidden_channels[2],
                out_channels=db_hidden_channels[3],
                kernel_size=4,
                strides=2,
                padding=1,
            )
        )
        self.main.append(
            D_block(
                in_channels=db_hidden_channels[3],
                out_channels=db_hidden_channels[4],
                kernel_size=4,
                strides=2,
                padding=1,
            )
        )
        self.main.append(
            nn.Conv1d(
                in_channels=db_hidden_channels[4],
                out_channels=1,
                kernel_size=4,
                bias=False,
            )
        )
        self.main.append(nn.Linear(13, 1))

        self.net = nn.Sequential(*self.main)

    def forward(self, h):
        input = h.unsqueeze(1)
        return self.net(input).squeeze(-1)


class Generator(nn.Module):
    """Generator network that combines latent vector and HLA sequence.

    Args:
        z_dim: Dimensionality of the latent vector.
        seq_len: Length of the peptide sequence.
        hla_len: Length of the HLA sequence.
        alphabet_size: Size of the amino acid alphabet.
        output_dim: Dimensionality of the output. Defaults to alphabet_size if None.
        embedding_dim: Dimensionality of the HLA embedding.
        hidden_size: Dimensionality of the hidden layers.
        num_map_layers: Number of layers in the mapping network.
    """

    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        hla_len: int,
        alphabet_size: int,
        output_dim: int | None = None,
        embedding_dim: int = 32,
        hidden_size: int = 256,
        num_map_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hla_len = hla_len
        self.alphabet_size = alphabet_size
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim or alphabet_size

        self.hla_flat_dim = self.hla_len * self.alphabet_size

        self.z_featurise = MappingNetwork(
            self.z_dim, self.hidden_size, num_layers=num_map_layers
        )

        self.hla_emb = nn.Embedding(self.alphabet_size, self.embedding_dim)

        self.hla_feat = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.hidden_size // 8, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 8, self.hidden_size // 4, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 4, self.hidden_size // 2, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 2, self.hidden_size, 3, 2),
        )

        self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.multi_conv_layers = ConvGenerator(hidden_size)

    def forward(self, z: Tensor, hla_idxs: Tensor) -> Tensor:
        z_feat = self.z_featurise(z.float())
        hla = self.hla_emb(hla_idxs)
        hla_feat = self.hla_feat(hla.transpose(1, 2)).squeeze(-1)
        h = torch.concat((z_feat, hla_feat), dim=-1)
        h = self.proj(h)

        h = self.multi_conv_layers(h, z_feat)
        return h


class Discriminator(nn.Module):
    """Discriminator network that combines peptide and HLA sequence.

    Args:
        seq_len: Length of the peptide sequence.
        hla_len: Length of the HLA sequence.
        alphabet_size: Size of the amino acid alphabet.
        input_dim: Dimensionality of the input. Defaults to alphabet_size if None.
        embedding_dim: Dimensionality of the HLA embedding.
        hidden_size: Dimensionality of the hidden layers.
    """

    def __init__(
        self,
        seq_len: int,
        hla_len: int,
        alphabet_size: int,
        input_dim: int | None = None,
        embedding_dim: int = 32,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hla_len = hla_len
        self.alphabet_size = alphabet_size
        self.input_dim = input_dim or alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.pep_scale = nn.Parameter(Tensor(self.input_dim).fill_(1.0))

        self.pep_featurise = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_size // 4, 3, 1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 4, self.hidden_size // 2, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 2, self.hidden_size, 3, 2),
        )

        self.hla_emb = nn.Embedding(self.alphabet_size, self.embedding_dim)

        self.hla_featurise = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.hidden_size // 8, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 8, self.hidden_size // 4, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 4, self.hidden_size // 2, 3, 2),
            nn.SiLU(),
            nn.Conv1d(self.hidden_size // 2, self.hidden_size, 3, 2),
        )

        self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.multi_conv_layers = ConvDiscriminator(hidden_size)

    def forward(self, peptide_1hot: Tensor, hla_idxs: Tensor) -> Tensor:
        pep = peptide_1hot * self.pep_scale
        hla = self.hla_emb(hla_idxs)

        pep_feat = self.pep_featurise(pep.transpose(1, 2)).squeeze(-1)
        hla_feat = self.hla_featurise(hla.transpose(1, 2)).squeeze(-1)

        h = torch.concat((pep_feat, hla_feat), dim=-1)
        h = self.proj(h)

        return self.multi_conv_layers(h)
