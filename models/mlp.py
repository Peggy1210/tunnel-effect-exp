from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Multi-layer perceptron with configurable depth and width."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 12,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        dims: List[int] = [input_dim] + [hidden_dim] * num_hidden_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(inplace=False),
                )
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for block in self.hidden_layers:
            linear = block[0]
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.classifier(x)

    @torch.no_grad()
    def get_representations(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Returns list of output from num_hidden_layers layers."""
        reps = OrderedDict()
        x = self.flatten(x)
        for idx, layer in enumerate(self.hidden_layers):
            x = layer(x)
            reps[f"hidden_layers.{idx}.0"] = x

        reps[f"classifier"] = self.classifier(x)
        return reps
    
    @torch.no_grad()
    def get_layer_names(self) -> List[str]:
        """Returns list of layer names in order."""
        names = []
        for idx, layer in enumerate(self.hidden_layers): # Hidden layers
            names.append(f"hidden_layers.{idx}.0")
        names.append(f"classifier") # Classifier
        return names
    
    @torch.no_grad()
    def get_layer_representations(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Returns the representation from a specific layer."""
        x = self.flatten(x)
        for idx, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if f"hidden_layers.{idx}.0" == layer_name:
                return x
            
        x = self.classifier(x)
        if f"classifier" == layer_name:
            return x

        raise ValueError(f"Layer not found: {layer_name}")
    

def build_mlp(model_name: str, num_classes: int) -> MLPClassifier:
    if model_name == "mlp6":
        return MLPClassifier(num_hidden_layers=6, num_classes=num_classes)
    elif model_name == "mlp8":
        return MLPClassifier(num_hidden_layers=8, num_classes=num_classes)
    elif model_name == "mlp10":
        return MLPClassifier(num_hidden_layers=10, num_classes=num_classes)
    elif model_name == "mlp12":
        return MLPClassifier(num_hidden_layers=12, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown MLP model: {model_name}")