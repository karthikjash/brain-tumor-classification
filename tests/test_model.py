import torch
from src.model import create_model

model = create_model()
dummy = torch.randn(1,3,224,224)
output = model(dummy)
print("Output shape: ", output.shape)

