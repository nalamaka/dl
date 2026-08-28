from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []
        curr_in = in_channels
        for _ in range(num_convs):
            layers.append(ConvBNReLU(curr_in, out_channels))
            curr_in = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []
        curr_in = in_channels
        for conv_idx in range(num_convs):
            target_channels = out_channels if conv_idx == num_convs - 1 else in_channels
            layers.append(ConvBNReLU(curr_in, target_channels))
            curr_in = target_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512, 512),
    ):
        super().__init__()
        block_depths = (2, 2, 3, 3, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        encoders = []
        in_channels = 3
        for out_channels, depth in zip(encoder_channels, block_depths):
            encoders.append(SegNetEncoderBlock(in_channels, out_channels, depth))
            in_channels = out_channels
        self.encoders = nn.ModuleList(encoders)

        decoder_specs = list(zip(reversed(encoder_channels), reversed((3, 3, 3, 2, 2))))
        self.decoders = nn.ModuleList()
        for idx, (in_channels, depth) in enumerate(decoder_specs):
            out_channels = 64 if idx == len(decoder_specs) - 1 else encoder_channels[-(idx + 2)]
            self.decoders.append(SegNetDecoderBlock(in_channels, out_channels, depth))

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_stack: list[torch.Tensor] = []
        size_stack: list[torch.Size] = []

        for encoder in self.encoders:
            x = encoder(x)
            size_stack.append(x.size())
            x, indices = self.pool(x)
            indices_stack.append(indices)

        for decoder in self.decoders:
            indices = indices_stack.pop()
            output_size = size_stack.pop()
            x = self.unpool(x, indices, output_size=output_size)
            x = decoder(x)

        return self.classifier(x)

