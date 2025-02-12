"""PyTorch model for the PaDiM model implementation."""



from random import sample
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch import nn

from hamacho.core.pre_processing import Tiler
from hamacho.plug_in.models.components import FeatureExtractor
from hamacho.plug_in.models.components import MultiVariateGaussian
from hamacho.plug_in.models.padim.anomaly_map import AnomalyMapGenerator

DIMS = {
    "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    "wide_resnet50_2": {"orig_dims": 1792, "reduced_dims": 550, "emb_scale": 4},
}


class PadimModel(nn.Module):
    """Padim Module.

    Args:
        input_size (Tuple[int, int]): Input size for the model.
        layers (List[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone(pretrained=True), layers=self.layers
        )
        self.dims = DIMS[backbone]
        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(
                sample(
                    range(0, DIMS[backbone]["orig_dims"]),
                    DIMS[backbone]["reduced_dims"],
                )
            ),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        n_features = DIMS[backbone]["reduced_dims"]
        patches_dims = torch.tensor(input_size) / DIMS[backbone]["emb_scale"]
        n_patches = patches_dims.ceil().prod().int().item()
        self.gaussian = MultiVariateGaussian(n_features, n_patches)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings,
                mean=self.gaussian.mean,
                inv_covariance=self.gaussian.inv_covariance,
            )

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(
                layer_embedding, size=embeddings.shape[-2:], mode="nearest"
            )
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
