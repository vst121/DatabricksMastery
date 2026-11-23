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

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# ImageNet preprocessing expected by pretrained ResNet18
preprocess = weights.transforms()

train_dataset = CIFAR10(
    root="/tmp/cifar10",
    train=True,
    download=True,
    transform=preprocess
)

test_dataset = CIFAR10(
    root="/tmp/cifar10",
    train=False,
    download=True,
    transform=preprocess
)

# CIFAR-10 labels
CAT = 3
DOG = 5

rng = np.random.default_rng(42)

# Select balanced training and validation indices
train_indices = []
val_indices = []

for label in [CAT, DOG]:
    indices = np.where(
        np.array(train_dataset.targets) == label
    )[0]

    rng.shuffle(indices)

    train_indices.extend(indices[:500])
    val_indices.extend(indices[500:600])

# Select balanced test indices
test_indices = []

for label in [CAT, DOG]:
    indices = np.where(
        np.array(test_dataset.targets) == label
    )[0]

    rng.shuffle(indices)
    test_indices.extend(indices[:100])

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)
test_subset = Subset(test_dataset, test_indices)

print("Training:", len(train_subset))
print("Validation:", len(val_subset))
print("Testing:", len(test_subset))

class BinaryCIFAR10(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, original_label = self.dataset[index]

        binary_label = 0 if original_label == CAT else 1

        return image, binary_label


train_data = BinaryCIFAR10(train_subset)
val_data = BinaryCIFAR10(val_subset)
test_data = BinaryCIFAR10(test_subset)

batch_size = 16

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

print("Training device:", device)

epochs = 5

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Validation Loss: {avg_val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(f"Test Accuracy: {test_accuracy:.4f}")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Cat", "Dog"]
    )
)

import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ResNet18: Cat vs Dog")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

import mlflow

model.eval()

input_example = next(iter(test_loader))[0][:2].to(device)

with torch.no_grad():
    output_example = model(input_example)

signature = infer_signature(
    input_example.cpu().numpy(),
    output_example.cpu().numpy()
)

with mlflow.start_run(run_name="ResNet18_Transfer_Learning"):

    mlflow.log_param("architecture", "ResNet18")
    mlflow.log_param("pretrained", True)
    mlflow.log_param("pretrained_weights", "ImageNet")
    mlflow.log_param("strategy", "Feature Extraction")
    mlflow.log_param("dataset", "CIFAR-10")
    mlflow.log_param("classes", "Cat, Dog")
    mlflow.log_param("training_samples", len(train_data))
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", 0.001)

    mlflow.log_metric(
        "test_accuracy",
        float(test_accuracy)
    )

    mlflow.log_metric(
        "final_train_loss",
        float(train_losses[-1])
    )

    mlflow.log_metric(
        "final_validation_loss",
        float(val_losses[-1])
    )

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="resnet18_transfer_learning",
        signature=signature,
        input_example=input_example.cpu().numpy()
    )

print("ResNet18 model successfully logged!")

