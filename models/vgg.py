from collections import OrderedDict
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn

VGG_CFG = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}

class VGGClassifier(nn.Module):
    """VGG architecture with configurable depth."""

    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        width_mult: float = 1.0,
        fc_dim: int = 512,
        use_batch_norm: bool = True, ## FIXME
    ) -> None:
        super().__init__()
        cfg = VGG_CFG[num_layers]
        self.conv_blocks = self._make_layers(cfg, width_mult=width_mult, use_batch_norm=use_batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        last_channels = int(512 * width_mult)
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(last_channels, fc_dim),
                nn.ReLU(inplace=False),
                # nn.Dropout(0.5), ## FIXME
            ),
            nn.Sequential(
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(inplace=False),
                # nn.Dropout(0.5), ## FIXME
            ),
            nn.Sequential(
                nn.Linear(fc_dim, num_classes),
            )
            
        ])
        self._init_weights()

    def _make_layers(
        self,
        cfg: List,
        width_mult: float = 1.0,
        use_batch_norm: bool = False,
    ) -> nn.Sequential:
        layers = nn.ModuleList()
        in_channels = 3
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(item * width_mult)
                if use_batch_norm:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=False),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=False)
                    ))
                in_channels = out_channels
        return layers

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        for linear in self.classifier:
            x = linear(x)

        return x

    @torch.no_grad()
    def get_layer_names(self) -> List[str]:
        """Returns list of layer names corresponding to the order of features returned by get_representations."""
        layer_names = []
        conv_idx = 0
        for block_id, block in enumerate(self.conv_blocks):
            # Only record names for convolutional blocks (skip pooling-only blocks).
            if isinstance(block, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in block):
                conv_idx += 1
                layer_names.append(f"conv_blocks.{block_id}.0")

        for fc_idx, linear in enumerate(self.classifier):
            layer_names.append(f"classifier.{fc_idx}.0")
        return layer_names

    @torch.no_grad()
    def get_layer_representations(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Returns output from a specific conv or fc layer."""
        conv_idx = 0
        for block_id, block in enumerate(self.conv_blocks):
            x = block(x)

            # Only record outputs for convolutional blocks (skip pooling-only blocks).
            if isinstance(block, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in block):
                conv_idx += 1
                if f"conv_blocks.{block_id}.0" == layer_name:
                    return x.flatten(1)

        # Check fully connected layers
        x = self.avgpool(x)
        x = x.flatten(1)

        for fc_idx, linear in enumerate(self.classifier):
            x = linear(x)
            if f"classifier.{fc_idx}.0" == layer_name:
                return x

        raise ValueError(f"Layer name {layer_name} not found")

    @torch.no_grad()
    def get_representations(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Returns list of output from all conv and fc layers."""
        feats = OrderedDict()

        conv_idx = 0
        for block_id, block in enumerate(self.conv_blocks):
            x = block(x)

            # Only record outputs for convolutional blocks (skip pooling-only blocks).
            if isinstance(block, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in block):
                conv_idx += 1
                feats[f"conv_blocks.{block_id}.0"] = x.flatten(1)

        x = self.avgpool(x)
        x = x.flatten(1)

        for fc_idx, linear in enumerate(self.classifier):
            x = linear(x)
            feats[f"classifier.{fc_idx}.0"] = x

        return feats

def build_vgg(model_name: str, num_classes: int) -> VGGClassifier:
    if model_name == "vgg11":
        return VGGClassifier(num_layers=11, num_classes=num_classes)
    elif model_name == "vgg13":
        return VGGClassifier(num_layers=13, num_classes=num_classes)
    elif model_name == "vgg16":
        return VGGClassifier(num_layers=16, num_classes=num_classes)
    elif model_name == "vgg19":
        return VGGClassifier(num_layers=19, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")