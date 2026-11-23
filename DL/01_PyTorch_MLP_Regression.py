import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mlflow
import mlflow.pytorch

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("PyTorch:", torch.__version__)
print("MLflow:", mlflow.__version__)
print("CUDA available:", torch.cuda.is_available())

np.random.seed(42)

X = np.random.uniform(-3, 3, size=(2000, 10))

y = (
    3 * np.sin(X[:, 0])
    + 2 * np.square(X[:, 1])
    - 1.5 * X[:, 2] * X[:, 3]
    + 0.5 * X[:, 4]
    + np.random.normal(0, 0.5, 2000)
)

print("Features:", X.shape)
print("Target:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

print("Training batches:", len(train_loader))

class RegressionMLP(nn.Module):

    def __init__(self, input_size=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = RegressionMLP()

print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

print("Loss function:", criterion)
print("Optimizer:", optimizer)

training_losses = []

for epoch in range(epochs):

    model.train()

    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        predictions = model(batch_X)

        # Calculate loss
        loss = criterion(predictions, batch_y)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    training_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.4f}"
        )

plt.figure(figsize=(9, 5))

plt.plot(training_losses)

plt.title("PyTorch Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

plt.show()

model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

y_actual = y_test.numpy()

rmse = np.sqrt(
    mean_squared_error(y_actual, predictions)
)

mae = mean_absolute_error(
    y_actual, predictions
)

r2 = r2_score(
    y_actual, predictions
)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")

plt.figure(figsize=(7, 7))

plt.scatter(
    y_actual,
    predictions,
    alpha=0.5
)

min_value = min(y_actual.min(), predictions.min())
max_value = max(y_actual.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()

from mlflow.models import infer_signature

model.eval()

input_example = X_test[:5].numpy()

with torch.no_grad():
    output_example = model(X_test[:5]).numpy()

signature = infer_signature(
    input_example,
    output_example
)

with mlflow.start_run(run_name="PyTorch_MLP_Regression"):

    mlflow.log_param("architecture", "MLP")
    mlflow.log_param("hidden_layers", "64,32")
    mlflow.log_param("activation", "ReLU")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 32)

    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("r2", float(r2))

    mlflow.log_metric(
        "final_training_loss",
        float(training_losses[-1])
    )

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="pytorch_mlp",
        signature=signature,
        input_example=input_example
    )

    print("PyTorch model successfully logged!")

    