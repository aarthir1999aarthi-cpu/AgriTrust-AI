
import tensorflow as tf
from pathlib import Path
import json
import matplotlib.pyplot as plt

print("=" * 60)
print("AGRITRUST AI - CNN MODEL BUILDING")
print("=" * 60)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset_split_v2" / "train"
VAL_DIR = BASE_DIR / "dataset_split_v2" / "validation"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS = 5

# Load class names
class_names_path = RESULTS_DIR / "class_names.json"

with open(class_names_path, "r") as file:
    class_names = json.load(file)

print("\nTotal Classes:", len(class_names))

# Load training dataset
print("\nLoading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

# Load validation dataset
print("\nLoading Validation Dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Verify class order
if train_ds.class_names != class_names:
    raise ValueError("Training class names do not match class_names.json")

if val_ds.class_names != class_names:
    raise ValueError("Validation class names do not match class_names.json")

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build CNN Model
print("\nBuilding CNN Model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Architecture:")
model.summary()

# Model checkpoint
model_path = RESULTS_DIR / "agrTrust_cnn_model.keras"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.00001,
    verbose=1
)

# CSV training log
csv_logger = tf.keras.callbacks.CSVLogger(
    RESULTS_DIR / "cnn_training_log.csv",
    append=False
)

# Train Model
print("\nStarting CNN Model Training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]
)

# Restore and save best model
model.save(model_path)

print("\nCNN MODEL SAVED SUCCESSFULLY!")
print("Model Location:", model_path)

# Save training history
history_path = RESULTS_DIR / "cnn_training_history.json"

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(history_path, "w") as file:
    json.dump(history_data, file, indent=4)

print("\nTraining History Saved:", history_path)

# Accuracy Graph
plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("AgriTrust AI - CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULTS_DIR / "cnn_accuracy.png")
plt.close()

# Loss Graph
plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("AgriTrust AI - CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULTS_DIR / "cnn_loss.png")
plt.close()

print("\nAccuracy Graph Saved: cnn_accuracy.png")
print("Loss Graph Saved: cnn_loss.png")

print("\n" + "=" * 60)
print("CNN MODEL TRAINING COMPLETED!")
print("=" * 60)