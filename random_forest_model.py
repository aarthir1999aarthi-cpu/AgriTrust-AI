# ============================================================
# AGRITRUST AI - RANDOM FOREST MODEL
# ============================================================

import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("=" * 60)
print("AGRITRUST AI - RANDOM FOREST TRAINING")
print("=" * 60)


# STEP 1: Load CNN features

data = np.load("results/cnn_extracted_features.npz")

X_train = data["train_features"]
y_train = data["train_labels"]

X_val = data["validation_features"]
y_val = data["validation_labels"]

X_test = data["test_features"]
y_test = data["test_labels"]

print("\nDataset Loaded Successfully")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# STEP 2: Train Random Forest configurations

print("\nTraining Random Forest Models...")

configs = [
    {"n_estimators": 100, "max_depth": 20},
    {"n_estimators": 150, "max_depth": 30},
    {"n_estimators": 200, "max_depth": None}
]

best_model = None
best_accuracy = -1
best_config = None

for config in configs:

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        predictions
    )

    print(
        f"Trees: {config['n_estimators']} | "
        f"Depth: {config['max_depth']} | "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy
        best_model = model
        best_config = config


# STEP 3: Evaluate best model

print("\nEvaluating Best Random Forest Model...")

test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


# STEP 4: Display results

print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print("Best Configuration:", best_config)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# STEP 5: Save model

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    best_model,
    "models/random_forest_model.joblib"
)


# STEP 6: Save evaluation results

with open(
    "results/random_forest_results.txt",
    "w"
) as file:

    file.write("AGRITRUST AI - RANDOM FOREST RESULTS\n")
    file.write("=" * 50 + "\n")

    file.write(f"Best Configuration: {best_config}\n")
    file.write(f"Validation Accuracy: {best_accuracy * 100:.2f}%\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision * 100:.2f}%\n")
    file.write(f"Recall: {recall * 100:.2f}%\n")
    file.write(f"F1 Score: {f1 * 100:.2f}%\n\n")

    file.write(report)


print("\nModel saved successfully!")
print("File: models/random_forest_model.joblib")

print("\nResults saved successfully!")
print("File: results/random_forest_results.txt")

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETED!")
print("=" * 60)