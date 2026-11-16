import numpy as np
import joblib
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
FEATURE_PATH = BASE_DIR / "results" / "cnn_extracted_features.npz"

PLOT_PATH = BASE_DIR / "results" / "shap_feature_importance.png"
REPORT_PATH = BASE_DIR / "results" / "shap_explanation_report.txt"

print("=" * 60)
print("AGRITRUST AI - SHAP EXPLAINABILITY")
print("=" * 60)

# STEP 1: Load Random Forest model
print("\nSTEP 1: Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully!")

# STEP 2: Load CNN extracted features
print("\nSTEP 2: Loading CNN features...")

data = np.load(FEATURE_PATH)

X_train = data["train_features"]
X_test = data["test_features"]
y_test = data["test_labels"]

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Features per image:", X_test.shape[1])

# STEP 3: Select a manageable sample
print("\nSTEP 3: Preparing SHAP samples...")

rng = np.random.default_rng(42)

background_indices = rng.choice(
    len(X_train),
    size=min(100, len(X_train)),
    replace=False
)

test_indices = rng.choice(
    len(X_test),
    size=min(50, len(X_test)),
    replace=False
)

background_data = X_train[background_indices]
explain_data = X_test[test_indices]

# STEP 4: Create Tree SHAP explainer
print("\nSTEP 4: Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

# STEP 5: Calculate SHAP values
print("\nSTEP 5: Calculating SHAP values...")

shap_values = explainer.shap_values(explain_data)

print("SHAP values calculated successfully!")

# STEP 6: Handle SHAP output formats
if isinstance(shap_values, list):

    # Older SHAP multi-class output
    importance_values = np.mean(
        np.stack(
            [np.abs(values) for values in shap_values],
            axis=0
        ),
        axis=(0, 1, 2)
    )

elif isinstance(shap_values, np.ndarray):

    if shap_values.ndim == 3:

        # Samples x Features x Classes
        importance_values = np.mean(
            np.abs(shap_values),
            axis=(0, 2)
        )

    elif shap_values.ndim == 2:

        # Samples x Features
        importance_values = np.mean(
            np.abs(shap_values),
            axis=0
        )

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

else:
    raise TypeError("Unsupported SHAP output format.")

# STEP 7: Find top important features
print("\nSTEP 6: Identifying important features...")

top_indices = np.argsort(importance_values)[::-1][:15]

# STEP 8: Create feature importance graph
print("\nSTEP 7: Creating SHAP graph...")

plt.figure(figsize=(10, 7))

plt.barh(
    [f"CNN Feature {i}" for i in top_indices[::-1]],
    importance_values[top_indices[::-1]]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("CNN Extracted Features")
plt.title("AgriTrust AI - SHAP Feature Importance")

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()

# STEP 9: Save explanation report
with open(REPORT_PATH, "w", encoding="utf-8") as file:

    file.write("AGRITRUST AI - SHAP EXPLAINABILITY REPORT\n")
    file.write("=" * 55 + "\n\n")

    file.write("Model: Random Forest\n")
    file.write("Input: 128 CNN-extracted features\n")
    file.write(f"Explained test samples: {len(explain_data)}\n\n")

    file.write("TOP 15 IMPORTANT CNN FEATURES\n")
    file.write("-" * 40 + "\n")

    for rank, index in enumerate(top_indices, start=1):

        file.write(
            f"{rank}. CNN Feature {index}: "
            f"{importance_values[index]:.8f}\n"
        )

    file.write("\nINTERPRETATION NOTE\n")
    file.write(
        "SHAP values describe how CNN feature dimensions "
        "influence Random Forest predictions.\n"
    )

    file.write(
        "These feature dimensions do not directly identify "
        "specific leaf pixels or disease symptoms.\n"
    )

print("\nSHAP explanation completed successfully!")
print("Graph saved to:", PLOT_PATH)
print("Report saved to:", REPORT_PATH)