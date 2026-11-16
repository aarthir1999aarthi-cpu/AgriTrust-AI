
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# AGRITRUST AI - CNN MODEL EVALUATION
# ==========================================

print("=" * 60)
print("AGRITRUST AI - CNN MODEL EVALUATION")
print("=" * 60)

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "dataset_split_v2" / "test"
MODEL_PATH = PROJECT_DIR / "results" / "agrTrust_cnn_model.keras"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# STEP 1: CHECK FILES

print("\nSTEP 1: Checking Required Files...")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

# STEP 2: LOAD CLASS NAMES

print("\nSTEP 2: Loading Class Names...")

with open(RESULTS_DIR / "class_names.json", "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Total Classes:", len(class_names))

# STEP 3: LOAD TEST DATASET

print("\nSTEP 3: Loading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

if test_ds.class_names != class_names:
    raise ValueError("Test dataset class order mismatch!")

# STEP 4: NORMALIZE IMAGES

print("\nSTEP 4: Normalizing Test Images...")

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_ds = test_ds.map(
    lambda images, labels:
    (normalization_layer(images), labels)
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# STEP 5: LOAD CNN MODEL

print("\nSTEP 5: Loading Trained CNN Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("CNN Model Loaded Successfully!")

# STEP 6: EVALUATE MODEL

print("\nSTEP 6: Evaluating CNN Model...")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

test_loss, test_accuracy = model.evaluate(test_ds)

# STEP 7: GENERATE PREDICTIONS

print("\nSTEP 7: Generating Predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# STEP 8: PERFORMANCE METRICS

print("\nSTEP 8: Calculating Performance Metrics...")

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

print("\nCNN PERFORMANCE RESULTS")
print("=" * 50)

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")

# STEP 9: CLASSIFICATION REPORT

print("\nSTEP 9: Classification Report...")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)

with open(
    RESULTS_DIR / "cnn_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# STEP 10: CONFUSION MATRIX

print("\nSTEP 10: Creating Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(20, 18))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=True,
    include_values=False
)

plt.title("AgriTrust AI - CNN Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cnn_confusion_matrix.png",
    dpi=300
)

plt.close()

# STEP 11: SAVE METRICS

metrics = {
    "model": "CNN",
    "total_classes": len(class_names),
    "total_test_images": int(len(y_true)),
    "test_accuracy": float(accuracy),
    "test_loss": float(test_loss),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open(
    RESULTS_DIR / "cnn_evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

# FINAL OUTPUT

print("\n" + "=" * 60)
print("CNN MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Test Accuracy:", f"{accuracy * 100:.2f}%")
print("Precision:", f"{precision * 100:.2f}%")
print("Recall:", f"{recall * 100:.2f}%")
print("F1 Score:", f"{f1 * 100:.2f}%")

print("\nSaved Files:")
print("1. cnn_classification_report.txt")
print("2. cnn_confusion_matrix.png")
print("3. cnn_evaluation_metrics.json")

print("=" * 60)
