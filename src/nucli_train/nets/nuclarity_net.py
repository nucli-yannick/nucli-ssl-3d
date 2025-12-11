from __future__ import annotations

import torch
import torch.nn as nn

import torch.nn.functional as F

import math

class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """

    def __init__(self, scale):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale**3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(
            batch_size,
            nOut,
            self.scale,
            self.scale,
            self.scale,
            in_depth,
            in_height,
            in_width,
        )

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class ConvBlock_3x3(nn.Module):
    """
    A convolutional block that can include convolution layers, dropout layers,
    batch normalization layers, and an activation function.

    Parameters:
    spatial_dims (int): The dimension of the convolution (2 or 3).
    in_channel (int): Number of input channels.
    out_channel (int): Number of output channels.
    batch_norm (bool): Whether to include batch normalization.
    activation (str): Activation function to use.
    dropout (tuple[bool, float]): Dropout inclusion flag and probability.
    kernel_size (int, optional): Size of the convolution kernel.
    stride (int, optional): Stride of the convolution.
    padding (int, optional): Padding added to all sides of the input.
    bias (bool, optional): If `True`, adds a learnable bias to the output.
    groups (int, optional): Number of blocked connections from input channels to output channels.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channel: int,
        out_channel: int,
        batch_norm: bool,
        activation: str,
        dropout: tuple[bool, float],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        groups: int = 1,
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        Dropout = nn.Dropout2d if spatial_dims == 2 else nn.Dropout3d

        layers = [
            Conv(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )
        ]
        if dropout[0]:
            layers.append(Dropout(dropout[1]))
        if batch_norm:
            layers.append(BatchNorm(out_channel))
        layers.append(getattr(nn, activation)())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock_3x3(nn.Module):
    """
    A double convolutional block consisting of two ConvBlock_3x3 modules.

    Parameters:
    spatial_dims (int): The dimension of the convolution (2 or 3).
    in_channel (int): Number of input channels for the first convolution.
    mid_channel (int): Number of output channels for the first convolution and input channels for the second convolution.
    out_channel (int): Number of output channels for the second convolution.
    batch_norm (bool): Whether to include batch normalization in both convolutions.
    activation (str): Activation function to use in both convolutions.
    dropout (tuple[bool, float]): Dropout inclusion flag and probability for both convolutions.
    groups (int, optional): Number of blocked connections from input channels to output channels in both convolutions.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channel: int,
        mid_channel: int,
        out_channel: int,
        batch_norm: bool,
        activation: str,
        dropout: tuple[bool, float],
        groups: int = 1,
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."

        self.conv1 = ConvBlock_3x3(
            spatial_dims,
            in_channel,
            mid_channel,
            batch_norm,
            activation,
            dropout,
            groups=groups,
        )
        self.conv2 = ConvBlock_3x3(
            spatial_dims,
            mid_channel,
            out_channel,
            batch_norm,
            activation,
            dropout,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channel,
        out_channel,
        pool_mode,
        batch_norm,
        activation,
        dropout,
    ):
        super().__init__()

        if pool_mode == "maxpool":
            pool_operation = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        elif pool_mode == "meanpool":
            pool_operation = nn.AvgPool2d if spatial_dims == 2 else nn.AvgPool3d
        else:
            raise NotImplementedError(f"{pool_mode} not implemented")

        double_conv = DoubleConvBlock_3x3(
            spatial_dims,
            in_channel,
            out_channel,
            out_channel,
            batch_norm,
            activation,
            dropout,
        )

        self.down = nn.Sequential(pool_operation(2), double_conv)

    def forward(self, x):
        return self.down(x)


class ResizePadding(nn.Module):
    """
    ResizePadding module for resizing and optionally concatenating feature maps with different dimensions.
    """

    def __init__(self, spatial_dims: int):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."
        self.spatial_dims = spatial_dims

    def forward(
        self, x_decode: torch.Tensor, x_encode: torch.Tensor, concatenate: bool
    ) -> torch.Tensor:
        """
        Resizes x_decode to match the dimensions of x_encode and optionally concatenates them along the channel dimension.

        Args:
            x_decode (torch.Tensor): The tensor to be resized.
            x_encode (torch.Tensor): The reference tensor for sizing.
            concatenate (bool): Whether to concatenate x_decode with x_encode.

        Returns:
            torch.Tensor: Resized and optionally concatenated tensor.
        """
        # Calculate the padding needed
        padding = [0, 0] * self.spatial_dims
        for i in range(self.spatial_dims):
            padding_diff = x_encode.size()[-(i + 1)] - x_decode.size()[-(i + 1)]
            padding[2 * i : 2 * i + 2] = [0, padding_diff]

        # Apply padding
        x_decode = F.pad(x_decode, padding)

        return torch.cat([x_decode, x_encode], dim=1) if concatenate else x_decode


class UpSample(nn.Module):
    """
    UpSample module for U-Net architecture, performing upsampling and optionally merging with a skip connection.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channel: int,
        mid_channel: int,
        out_channel: int,
        batch_norm: bool,
        activation: str,
        dropout: tuple[bool, float],
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."

        if in_channel == mid_channel:  # remove skip connection
            mid_channel = out_channel
            self.skip_connection = False
        else:
            self.skip_connection = True

        mode = "bilinear" if spatial_dims == 2 else "trilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

        self.resize = ResizePadding(spatial_dims)

        self.double_conv = DoubleConvBlock_3x3(
            spatial_dims,
            in_channel,
            mid_channel,
            out_channel,
            batch_norm,
            activation,
            dropout,
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x = (
            self.resize(x, x_skip, concatenate=self.skip_connection)
            if self.skip_connection
            else x
        )
        x = self.double_conv(x)
        return x


class UpShuffle(nn.Module):
    """
    UpShuffle module for the U-Net architecture, performing upsampling using pixel shuffling
    and optionally merging with a skip connection.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channel: int,
        mid_channel: int,
        out_channel: int,
        batch_norm: bool,
        activation: str,
        dropout: tuple[bool, float],
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        shuffle = nn.PixelShuffle if spatial_dims == 2 else PixelShuffle3d
        times = 4 if spatial_dims == 2 else 8  # 2^2 for 2D and 2^3 for 3D

        if in_channel == mid_channel:  # remove skip connection
            mid_channel = out_channel
            self.skip_connection = False
        else:
            self.skip_connection = True

        self.up = nn.Sequential(
            Conv(
                in_channels=in_channel, out_channels=out_channel * times, kernel_size=1
            ),
            getattr(nn, activation)(inplace=True),
            shuffle(2),
        )

        self.resize = ResizePadding(spatial_dims)

        self.double_conv = DoubleConvBlock_3x3(
            spatial_dims,
            in_channel,
            mid_channel,
            out_channel,
            batch_norm,
            activation,
            dropout,
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x = (
            self.resize(x, x_skip, concatenate=self.skip_connection)
            if self.skip_connection
            else x
        )
        x = self.double_conv(x)
        return x


class UpConv(nn.Module):
    """
    UpConv module for the U-Net architecture, performing up-convolution and merging with the skip connection.
    """

    def __init__(
        self,
        spatial_dims: int,
        main_channel: int,
        skip_channel: int,
        main_channel_next: int,
        batch_norm: bool,
        activation: str,
        dropout: tuple[bool, float],
    ):
        super().__init__()

        ConvTranspose = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        self.up = ConvTranspose(main_channel, main_channel, kernel_size=2, stride=2)

        self.resize = ResizePadding(spatial_dims)

        mid_channel, out_channel = (
            (main_channel, skip_channel)
            if skip_channel != 0
            else (main_channel_next, main_channel_next)
        )
        self.skip_connection = skip_channel != 0

        self.double_conv = DoubleConvBlock_3x3(
            spatial_dims,
            main_channel + skip_channel,
            mid_channel,
            out_channel,
            batch_norm,
            activation,
            dropout,
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x = (
            self.resize(x, x_skip, concatenate=self.skip_connection)
            if self.skip_connection
            else x
        )
        x = self.double_conv(x)
        return x





class OutConv_1x1(nn.Module):
    """
    OutConv module for the U-Net architecture, performing 1x1 convolution to map to the desired number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channel: int, out_channel: int, groups: int = 1
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "spatial_dims must be either 2 or 3."
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv = Conv(in_channel, out_channel, kernel_size=1, groups=groups)

    def forward(self, x):
        return self.conv(x)


class NucUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        features_main: list = [16, 32, 64, 128, 256],
        features_skip: list = [16, 32, 64, 128],
        up_mode: str = "upshuffle",
        pool_mode: str = "meanpool",
        batch_norm: bool = True,
        activation: str = "ReLU",
        residual_connection: bool = True,
        ReLU_OUT: bool = True,
        tanh_OUT: bool = False,
        dropout: tuple[bool, float] = (False, 0.2),
    ):
        super().__init__()
        self.__name__ = "UNet"
        self.params = locals()
        self.params.pop("self")
        self.params.pop("__class__")

        assert spatial_dims in [2, 3], "Only 2D or 3D arrays are supported."
        self.spatial_dims = spatial_dims
        self.depth = len(features_main) - 1
        self.residual_ch = in_channels // 2 if residual_connection else None

        self.resizing = ResizePadding(spatial_dims)

        # INCOME, FIRST LAYERS
        self.income = DoubleConvBlock_3x3(
            spatial_dims,
            in_channels,
            features_main[0],
            features_main[0],
            batch_norm,
            activation,
            dropout,
        )

        # DOWN PART
        self.Downs = nn.ModuleList(
            [
                DownBlock(
                    spatial_dims,
                    features_main[i],
                    features_main[i + 1],
                    pool_mode,
                    batch_norm,
                    activation,
                    dropout,
                )
                for i in range(self.depth)
            ]
        )

        if features_skip is None:
            features_skip = [0] * self.depth

        # UP PART
        if up_mode == "upsample":
            self.Ups = nn.ModuleList(
                [
                    UpSample(
                        spatial_dims,
                        features_main[i + 1] + features_skip[i],
                        features_main[i + 1],
                        features_main[i],
                        batch_norm,
                        activation,
                        dropout,
                    )
                    for i in reversed(range(self.depth))
                ]
            )

        elif up_mode == "upconv":
            self.Ups = nn.ModuleList(
                [
                    UpConv(
                        spatial_dims,
                        features_main[i + 1],
                        features_skip[i],
                        features_main[i],
                        batch_norm,
                        activation,
                        dropout,
                    )
                    for i in reversed(range(self.depth))
                ]
            )
        elif up_mode == "upshuffle":
            self.Ups = nn.ModuleList(
                [
                    UpShuffle(
                        spatial_dims,
                        features_main[i + 1],
                        features_skip[i],
                        features_main[i],
                        batch_norm,
                        activation,
                        dropout,
                    )
                    for i in reversed(range(self.depth))
                ]
            )
        else:
            raise NotImplementedError(f"Requested up_mode {up_mode} is not valid.")

        # OUT
        self.outcome = OutConv_1x1(spatial_dims, features_main[0], out_channels)

        # ADD RELU AT THE END
        if ReLU_OUT:
            self.activ_OUT = getattr(nn, "ReLU")()
        elif tanh_OUT:
            self.activ_OUT = getattr(nn, "Tanh")()
        else:
            self.activ_OUT = None

    def model_hyperparams(self) -> dict:
        """Return the hyperparameters of the model."""
        return self.params

    def decode(self, x):
        return self.Downs(x)


    def forward(self, x):
        residual_x = x
        x = self.income(x)
        save_skip = []

        for encode_block in self.Downs:
            save_skip.append(x)
            x = encode_block(x)

        for decode_block in self.Ups:
            x = decode_block(x, save_skip.pop())

        x = self.outcome(x)

        if self.residual_ch is not None:
            residual_slice = slice(self.residual_ch, self.residual_ch + 1)
            y = residual_x[:, residual_slice, ...]
            x = self.resizing(x, y, concatenate=False) + y
        else:
            x = self.resizing(x, residual_x, concatenate=False)

        return self.activ_OUT(x) if self.activ_OUT else x


    def get_optimizer(self):
        k = 1/16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=math.sqrt(k)*1e-4
        )
        return optimizer

    def load_checkpoint(self, p, _):
        old_state = torch.load(p, weights_only=False)['state_dict']
        new_state_d = {}

        for k, v in old_state.items():
            if 'perceptual' in k:
                continue
            if k.startswith('model.'):
                new_state_d[k[6:]] = v
            else:
                new_state_d[k] = v


        self.load_state_dict(new_state_d)