import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("PyTorch:", torch.__version__)

digits = load_digits()

X = digits.images
y = digits.target

print("Images:", X.shape)
print("Labels:", y.shape)
print("Classes:", np.unique(y))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Normalize pixels to [0, 1]
X_train = X_train.astype(np.float32) / 16.0
X_val = X_val.astype(np.float32) / 16.0
X_test = X_test.astype(np.float32) / 16.0

# Add channel dimension: (N, 1, 8, 8)
X_train = X_train[:, np.newaxis, :, :]
X_val = X_val[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
X_val_t = torch.tensor(X_val)
X_test_t = torch.tensor(X_test)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print("Training:", X_train_t.shape)
print("Validation:", X_val_t.shape)
print("Testing:", X_test_t.shape)

batch_size = 32

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=batch_size,
    shuffle=False
)

class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = DigitCNN()

print(model)

total_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", total_params)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    running_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()

    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            running_val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

model.eval()

with torch.no_grad():
    test_logits = model(X_test_t)
    test_predictions = torch.argmax(test_logits, dim=1)

y_pred = test_predictions.numpy()

test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("CNN Confusion Matrix")

plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

import mlflow
from mlflow.models import infer_signature

model.eval()

input_example = X_test[:5]

with torch.no_grad():
    output_example = model(
        torch.tensor(input_example)
    ).numpy()

signature = infer_signature(
    input_example,
    output_example
)

with mlflow.start_run(run_name="PyTorch_CNN_Digit_Classification"):

    mlflow.log_param("architecture", "CNN")
    mlflow.log_param("input_shape", "1x8x8")
    mlflow.log_param("conv_layers", "16,32")
    mlflow.log_param("hidden_layers", "64")
    mlflow.log_param("activation", "ReLU")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

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
        name="pytorch_cnn_digits",
        signature=signature,
        input_example=input_example
    )

print("CNN model successfully logged!")

