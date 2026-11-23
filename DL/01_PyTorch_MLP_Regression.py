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