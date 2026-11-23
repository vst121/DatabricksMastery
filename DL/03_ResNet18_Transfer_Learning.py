import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

print(model)

print(model.fc)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel() for p in model.parameters()
)

print("Trainable parameters:", trainable)
print("Total parameters:", total)
print("Trainable percentage:", round(trainable / total * 100, 2))

