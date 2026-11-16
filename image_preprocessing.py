import tensorflow as tf
import os
import json
from pathlib import Path

# ==========================================
# AGRITRUST AI - IMAGE PREPROCESSING
# ==========================================

print("=" * 60)
print("AGRITRUST AI - IMAGE PREPROCESSING")
print("=" * 60)

# 1. CONFIGURATION

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset_split_v2"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# 2. CHECK DATASET PATH

print("\nSTEP 1: Checking Dataset...")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

train_path = DATASET_DIR / "train"
validation_path = DATASET_DIR / "validation"
test_path = DATASET_DIR / "test"

for path in [train_path, validation_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {path}"
        )

print("Dataset folders verified successfully!")

# 3. LOAD TRAINING DATASET

print("\nSTEP 2: Loading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

# 4. LOAD VALIDATION DATASET

print("\nSTEP 3: Loading Validation Dataset...")

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# 5. LOAD TEST DATASET

print("\nSTEP 4: Loading Testing Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# 6. CHECK CLASS NAMES

print("\nSTEP 5: Checking Disease Categories...")

class_names = train_ds.class_names

if len(class_names) != 38:
    raise ValueError(
        f"Expected 38 categories, found {len(class_names)}"
    )

if class_names != validation_ds.class_names:
    raise ValueError("Training and validation classes differ!")

if class_names != test_ds.class_names:
    raise ValueError("Training and testing classes differ!")

print("Total Categories:", len(class_names))

for index, name in enumerate(class_names):
    print(f"{index + 1}. {name}")

# 7. IMAGE NORMALIZATION

print("\nSTEP 6: Normalizing Images...")

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=AUTOTUNE
)

validation_ds = validation_ds.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=AUTOTUNE
)

# 8. PREFETCH DATASETS

print("\nSTEP 7: Optimizing Dataset Loading...")

train_ds = train_ds.prefetch(AUTOTUNE)

validation_ds = validation_ds.prefetch(AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)

# 9. SAVE CLASS NAMES

results_dir = PROJECT_DIR / "results"

results_dir.mkdir(exist_ok=True)

class_file = results_dir / "class_names.json"

with open(class_file, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# 10. VERIFY IMAGE BATCH

print("\nSTEP 8: Verifying Image Batch...")

for images, labels in train_ds.take(1):

    print("Image Shape:", images.shape)

    print("Label Shape:", labels.shape)

    print("Minimum Pixel Value:",
          float(tf.reduce_min(images)))

    print("Maximum Pixel Value:",
          float(tf.reduce_max(images)))

# 11. FINAL OUTPUT

print("\n" + "=" * 60)

print("IMAGE PREPROCESSING COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("Image Size:", IMAGE_SIZE)

print("Batch Size:", BATCH_SIZE)

print("Total Categories:", len(class_names))

print("Normalization: 0 to 1")

print("Training Dataset: READY")

print("Validation Dataset: READY")

print("Testing Dataset: READY")

print("Class Names Saved:", class_file)

print("=" * 60)