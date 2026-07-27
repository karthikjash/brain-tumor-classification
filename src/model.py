import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
)

from src.config import NUM_CLASSES

def create_model():

    model = efficientnet_b0(
        weights = EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features,NUM_CLASSES),

    )
    return model