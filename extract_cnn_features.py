import os
import json
import numpy as np
import tensorflow as tf

print("=" * 60)
print("AGRITRUST AI - CNN FEATURE EXTRACTION")
print("=" * 60)

# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = os.getcwd()

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset_split_v2"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "results", "agrTrust_cnn_model.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results"
)

CLASS_NAMES_PATH = os.path.join(
    OUTPUT_DIR, "class_names.json"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# STEP 2: CHECK REQUIRED FILES
# ==========================================

print("\nSTEP 1: Checking required files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

print("All required files found!")

# ==========================================
# STEP 3: LOAD TRAINED CNN MODEL
# ==========================================

print("\nSTEP 2: Loading trained CNN model...")

cnn_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("CNN model loaded successfully!")

# Initialize Sequential model
dummy_input = tf.zeros(
    (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    dtype=tf.float32
)

_ = cnn_model(dummy_input, training=False)

print("CNN model initialized successfully!")

# ==========================================
# STEP 4: CREATE FEATURE EXTRACTOR
# ==========================================

print("\nSTEP 3: Creating feature extractor...")

try:

    dense_layer = cnn_model.get_layer("dense")

except ValueError:

    print("\nAvailable model layers:")

    for layer in cnn_model.layers:
        print(layer.name)

    raise ValueError(
        "Dense layer named 'dense' was not found. "
        "Check the CNN model architecture."
    )

feature_extractor = tf.keras.Model(
    inputs=cnn_model.inputs,
    outputs=dense_layer.output
)

print("Feature extractor created successfully!")
print("Feature dimensions:", dense_layer.units)

# ==========================================
# STEP 5: LOAD CLASS NAMES
# ==========================================

print("\nSTEP 4: Loading class names...")

with open(CLASS_NAMES_PATH, "r") as file:

    saved_class_names = json.load(file)

print("Total classes:", len(saved_class_names))

# ==========================================
# STEP 6: EXTRACT FEATURES
# ==========================================

all_features = {}

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 60)

    print("PROCESSING:", split.upper())

    print("=" * 60)

    split_path = os.path.join(
        DATASET_DIR, split
    )

    if not os.path.exists(split_path):

        raise FileNotFoundError(
            f"{split} folder not found!"
        )

    # Load image dataset

    dataset = tf.keras.utils.image_dataset_from_directory(

        split_path,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False

    )

    class_names = dataset.class_names

    # Verify class names

    if class_names != saved_class_names:

        raise ValueError(
            f"Class names mismatch in {split} dataset!"
        )

    print("Number of classes:", len(class_names))

    # Normalize images

    dataset = dataset.map(

        lambda images, labels: (

            tf.cast(images, tf.float32) / 255.0,

            labels

        )

    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    feature_batches = []

    label_batches = []

    print("\nExtracting CNN features...")

    # Extract features batch by batch

    for images, labels in dataset:

        features = feature_extractor(

            images,

            training=False

        )

        feature_batches.append(
            features.numpy()
        )

        label_batches.append(
            labels.numpy()
        )

    # Combine extracted features

    X = np.concatenate(
        feature_batches,
        axis=0
    )

    y = np.concatenate(
        label_batches,
        axis=0
    )

    # Store features

    all_features[split + "_features"] = X

    all_features[split + "_labels"] = y

    print("\nExtraction completed!")

    print("Images processed:", len(y))

    print("Feature shape:", X.shape)

    print("Label shape:", y.shape)

    # Clear memory

    del dataset

    del feature_batches

    del label_batches

# ==========================================
# STEP 7: SAVE EXTRACTED FEATURES
# ==========================================

print("\nSTEP 5: Saving extracted features...")

output_path = os.path.join(

    OUTPUT_DIR,

    "cnn_extracted_features.npz"

)

np.savez_compressed(

    output_path,

    **all_features

)

# ==========================================
# STEP 8: FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)

print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("Total Classes:", len(saved_class_names))

print("Feature Dimensions:", dense_layer.units)

print(
    "Training Features:",
    all_features["train_features"].shape
)

print(
    "Validation Features:",
    all_features["validation_features"].shape
)

print(
    "Testing Features:",
    all_features["test_features"].shape
)

print("\nSaved File:")

print("results/cnn_extracted_features.npz")

print("=" * 60)