import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
FEATURE_PATH = BASE_DIR / "results" / "cnn_extracted_features.npz"
OUTPUT_PATH = BASE_DIR / "results" / "reliability_results.txt"

TARGET_PRECISION = 0.95

print("=" * 55)
print("AGRITRUST AI - SELF RELIABILITY DETECTION")
print("=" * 55)

# STEP 1: Load Random Forest model
print("\nSTEP 1: Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully!")

# STEP 2: Load extracted CNN features
print("\nSTEP 2: Loading CNN features...")

data = np.load(FEATURE_PATH)

X_val = data["validation_features"]
y_val = data["validation_labels"]

X_test = data["test_features"]
y_test = data["test_labels"]

print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# STEP 3: Get validation predictions and confidence scores
print("\nSTEP 3: Analyzing validation predictions...")

val_probabilities = model.predict_proba(X_val)

val_predictions = model.classes_[
    np.argmax(val_probabilities, axis=1)
]

val_confidence = np.max(val_probabilities, axis=1)

val_correct = (val_predictions == y_val)

# STEP 4: Select threshold using validation data only
print("\nSTEP 4: Selecting reliability threshold...")

candidate_thresholds = np.unique(val_confidence)

best_threshold = None
best_precision = 0
best_coverage = 0

for threshold in candidate_thresholds:

    accepted = val_confidence >= threshold

    if np.sum(accepted) == 0:
        continue

    precision = np.mean(val_correct[accepted])
    coverage = np.mean(accepted)

    if precision >= TARGET_PRECISION and coverage > best_coverage:
        best_threshold = threshold
        best_precision = precision
        best_coverage = coverage

# Fallback if target precision is not achieved
if best_threshold is None:

    print("Warning: Target precision was not achieved.")

    for threshold in candidate_thresholds:

        accepted = val_confidence >= threshold

        if np.sum(accepted) == 0:
            continue

        precision = np.mean(val_correct[accepted])
        coverage = np.mean(accepted)

        if (
            precision > best_precision
            or (
                precision == best_precision
                and coverage > best_coverage
            )
        ):
            best_threshold = threshold
            best_precision = precision
            best_coverage = coverage

print("Selected confidence threshold:", round(float(best_threshold), 4))
print("Validation accepted precision:", round(best_precision * 100, 2), "%")
print("Validation coverage:", round(best_coverage * 100, 2), "%")

# STEP 5: Evaluate on unseen test data
print("\nSTEP 5: Evaluating reliability on test data...")

test_probabilities = model.predict_proba(X_test)

test_predictions = model.classes_[
    np.argmax(test_probabilities, axis=1)
]

test_confidence = np.max(test_probabilities, axis=1)

test_correct = (test_predictions == y_test)

accepted = test_confidence >= best_threshold

accepted_count = int(np.sum(accepted))
uncertain_count = int(len(X_test) - accepted_count)

coverage = np.mean(accepted) * 100

if accepted_count > 0:
    reliable_accuracy = (
        np.mean(test_correct[accepted]) * 100
    )
else:
    reliable_accuracy = 0

overall_accuracy = accuracy_score(
    y_test,
    test_predictions
) * 100

# STEP 6: Display results
print("\n" + "=" * 55)
print("SELF RELIABILITY DETECTION RESULTS")
print("=" * 55)

print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"Confidence Threshold: {best_threshold:.4f}")
print(f"Accepted Predictions: {accepted_count}")
print(f"Uncertain Predictions: {uncertain_count}")
print(f"Coverage: {coverage:.2f}%")
print(f"Accuracy of Accepted Predictions: {reliable_accuracy:.2f}%")

# STEP 7: Save results
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    file.write("AGRITRUST AI - SELF RELIABILITY DETECTION\n")
    file.write("=" * 50 + "\n")

    file.write(f"Overall Test Accuracy: {overall_accuracy:.2f}%\n")
    file.write(f"Confidence Threshold: {best_threshold:.4f}\n")
    file.write(f"Accepted Predictions: {accepted_count}\n")
    file.write(f"Uncertain Predictions: {uncertain_count}\n")
    file.write(f"Coverage: {coverage:.2f}%\n")
    file.write(
        f"Accepted Prediction Accuracy: {reliable_accuracy:.2f}%\n"
    )

print("\nResults saved to:", OUTPUT_PATH)
print("\nReliability detection completed!")