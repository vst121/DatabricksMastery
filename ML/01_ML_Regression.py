import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import mlflow
import mlflow.sklearn

print("Libraries imported successfully!")
print("MLflow version:", mlflow.__version__)

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=15,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(10)]

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("Dataset shape:", df.shape)
display(df.head())

display(df.describe().T)

plt.figure(figsize=(8, 4))
plt.hist(df["target"], bins=30)
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.show()

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.4f}")

results_df = pd.DataFrame(results)
display(results_df)

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE (lower is better)")
plt.show()

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        mae = mean_absolute_error(
            y_test, predictions
        )

        r2 = r2_score(
            y_test, predictions
        )

        mlflow.log_param("model_type", name)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        if name == "Random Forest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"Logged experiment: {name}")