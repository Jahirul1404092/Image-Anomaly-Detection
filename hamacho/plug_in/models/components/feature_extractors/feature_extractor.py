"""Feature Extractor.

This script extracts features from a CNN network
"""



from typing import Callable
from typing import Dict
from typing import Iterable

import torch
from torch import Tensor
from torch import nn


class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.

    Example:
        >>> import torch
        >>> import torchvision
        >>> from hamacho.plug_in.models import FeatureExtractor

        >>> model = FeatureExtractor(model=torchvision.models.resnet18(), layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}
        self.out_dims = []

        for layer_id in layers:
            layer = dict([*self.backbone.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))
            # get output dimension of features if available
            layer_modules = [*layer.modules()]
            for idx in reversed(range(len(layer_modules))):
                if hasattr(layer_modules[idx], "out_channels"):
                    self.out_dims.append(layer_modules[idx].out_channels)
                    break

    def get_features(self, layer_id: str) -> Callable:
        """Get layer features.

        Args:
            layer_id (str): Layer ID

        Returns:
            Layer features
        """

        def hook(_, __, output):
            """Hook to extract features via a forward-pass.

            Args:
              output: Feature map collected after the forward-pass.
            """
            self._features[layer_id] = output

        return hook

    def forward(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(input_tensor)
        return self._features
