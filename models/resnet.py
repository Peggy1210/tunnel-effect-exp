

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

BLOCK_CONFIGS = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
}

class ResNetClassifier(nn.Module):
    """ResNet architecture with configurable depth."""

    def __init__(self, num_layers: int, num_classes: int = 10, width_multiplier: float = 1.0) -> None:
        super().__init__()
        self.inplanes = 64

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        layers = BLOCK_CONFIGS[num_layers]
        self.layer1 = self._make_layer(64, layers[0], stride=1, width_mult=width_multiplier)
        self.layer2 = self._make_layer(128, layers[1], stride=2, width_mult=width_multiplier)
        self.layer3 = self._make_layer(256, layers[2], stride=2, width_mult=width_multiplier)
        self.layer4 = self._make_layer(512, layers[3], stride=2, width_mult=width_multiplier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_multiplier), num_classes)

        self._init_weights()

    def _make_layer(self, channel: int, num_blocks: int, stride: int, width_mult: float) -> nn.Sequential:
        layers = []
        channels = int(channel * width_mult)

        # First block with potential stride
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

        layers.append(BasicBlock(self.inplanes, channels, stride, downsample))
        self.inplanes = channels

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.inplanes, channels, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
    
    @torch.no_grad()
    def get_layer_names(self) -> List[str]:
        """Returns list of layer names corresponding to the order of features returned by get_representations."""
        layer_names = []
        for stage_id, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for block_id, block in enumerate(stage):
                layer_names.append(f"layer{stage_id + 1}.{block_id}.conv2")
        return layer_names

    @torch.no_grad()
    def get_layer_representations(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Returns output from a specific layer (after conv2+bn2, before residual).
        Layer names are in format: "layer{stage}.{block}.conv2"
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for stage_id, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for block_id, block in enumerate(stage):
                if f"layer{stage_id + 1}.{block_id}.conv2" == layer_name:
                    # Extract output after conv2+bn2 (before residual connection)
                    x = block.forward_until_conv2(x)
                    return x.flatten(1)
                x = block(x)
        raise ValueError(f"Layer name {layer_name} not found")

    @torch.no_grad()
    def get_representations(self, x: torch.Tensor) -> OrderedDict:
        """
        Returns output from all conv2 layers (after conv2+bn2, before residual connection).
        This ensures consistent feature extraction across all blocks.
        """
        feats = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for stage_id, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for block_id, block in enumerate(stage):
                # Extract after conv2+bn2 (before residual)
                x_conv2 = block.forward_until_conv2(x)
                feats[f"layer{stage_id + 1}.{block_id}.conv2"] = x_conv2.flatten(1)
                # Update x for next block (with residual connection)
                x = block(x)
        return feats


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_until_conv2(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output after conv2+bn2, before residual connection."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out


def build_resnet(model_name: str, num_classes: int) -> ResNetClassifier:
    if model_name == "resnet18":
        return ResNetClassifier(num_layers=18, num_classes=num_classes, width_multiplier=1.0)
    elif model_name == "resnet34":
        return ResNetClassifier(num_layers=34, num_classes=num_classes, width_multiplier=1.0)
    else:
        raise ValueError("Invalid model name")
    