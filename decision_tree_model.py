# ============================================================
# AGRITRUST AI - DECISION TREE MODEL
# ============================================================

import numpy as np
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("=" * 60)
print("AGRITRUST AI - DECISION TREE MODEL TRAINING")
print("=" * 60)


# STEP 1: Load extracted CNN features

data = np.load("results/cnn_extracted_features.npz")

X_train = data["train_features"]
y_train = data["train_labels"]

X_val = data["validation_features"]
y_val = data["validation_labels"]

X_test = data["test_features"]
y_test = data["test_labels"]

print("\nSTEP 1: Dataset Loaded")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
print("Features per image:", X_train.shape[1])


# STEP 2: Train Decision Tree models

print("\nSTEP 2: Training Decision Tree Models")

depth_values = [10, 20, 30, None]

best_model = None
best_accuracy = -1
best_depth = None

for depth in depth_values:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    print(
        f"Max Depth: {depth} | "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy
        best_model = model
        best_depth = depth


print("\nBest Decision Tree Model Selected")
print("Best Depth:", best_depth)
print("Best Validation Accuracy:", round(best_accuracy * 100, 2), "%")


# STEP 3: Evaluate on test dataset

print("\nSTEP 3: Testing Decision Tree Model")

test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

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
print("DECISION TREE MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:\n")

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

print(report)


# STEP 5: Save trained model

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    best_model,
    "models/decision_tree_model.joblib"
)


# STEP 6: Save evaluation results

with open(
    "results/decision_tree_results.txt",
    "w"
) as file:

    file.write("AGRITRUST AI - DECISION TREE RESULTS\n")
    file.write("=" * 50 + "\n")

    file.write(f"Best Depth: {best_depth}\n")
    file.write(f"Validation Accuracy: {best_accuracy * 100:.2f}%\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision * 100:.2f}%\n")
    file.write(f"Recall: {recall * 100:.2f}%\n")
    file.write(f"F1 Score: {f1 * 100:.2f}%\n\n")

    file.write(report)


print("\nModel saved successfully!")
print("File: models/decision_tree_model.joblib")

print("\nResults saved successfully!")
print("File: results/decision_tree_results.txt")

print("\n" + "=" * 60)
print("DECISION TREE TRAINING COMPLETED!")
print("=" * 60)