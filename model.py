"""
Hybrid Quantum-Driven Learning Framework for TB Detection.
Classical backbone (EfficientNet) + Quantum-Inspired (QI) layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CLASSES, BACKBONE_NAME, QUANTUM_INSPIRED_DIM, DROPOUT


class QuantumInspiredLayer(nn.Module):
    """
    Quantum-inspired layer: simulates amplitude encoding, parameterized rotations,
    and measurement. Maps feature vector to unit sphere (amplitude encoding),
    applies learned 'rotation' (orthogonal-like transform), then projection to logits.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc_encode = nn.Linear(in_features, hidden_dim)
        self.fc_rotate = nn.Linear(hidden_dim, hidden_dim)
        self.fc_measure = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_orthogonal()

    def _init_orthogonal(self):
        nn.init.orthogonal_(self.fc_rotate.weight)
        nn.init.zeros_(self.fc_rotate.bias)
        nn.init.xavier_uniform_(self.fc_encode.weight)
        nn.init.zeros_(self.fc_encode.bias)
        nn.init.xavier_uniform_(self.fc_measure.weight)
        nn.init.zeros_(self.fc_measure.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: project to hidden dim
        x = F.relu(self.fc_encode(x))
        x = self.dropout(x)
        # Amplitude encoding: L2 normalize (unit sphere - quantum state analogy)
        x = F.normalize(x, p=2, dim=1)
        # Parameterized "rotation" (orthogonal transform preserves norm)
        x = self.fc_rotate(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Measurement: project to class logits
        logits = self.fc_measure(x)
        return logits


class HybridQuantumTBClassifier(nn.Module):
    """
    Hybrid Quantum-Driven TB Classifier:
    - Backbone: EfficientNet (pretrained) for feature extraction
    - Quantum-Inspired head: amplitude encoding + rotation + measurement
    """
    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        num_classes: int = NUM_CLASSES,
        qi_dim: int = QUANTUM_INSPIRED_DIM,
        dropout: float = DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feat_dim = 1280
        elif backbone_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b1(weights=weights)
            feat_dim = 1280
        elif backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            feat_dim = 1536
        elif backbone_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            feat_dim = 1792
        else:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feat_dim = 1280

        # Keep parts accessible for Grad-CAM hooks
        self.backbone_features = backbone.features
        self.backbone_pool = backbone.avgpool
        self.qi_head = QuantumInspiredLayer(feat_dim, qi_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone_pool(self.backbone_features(x))
        features = torch.flatten(features, 1)
        return self.qi_head(features)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone_features.parameters():
            p.requires_grad = not freeze


def build_model(pretrained: bool = True) -> nn.Module:
    return HybridQuantumTBClassifier(pretrained=pretrained)
