import os
import sys
import json
import traceback

import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, request, jsonify, send_file
from PIL import Image


# ==================================================
# 1. PROJECT PATH
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Add project root to Python path
sys.path.insert(0, BASE_DIR)

from database.database import (
    save_prediction,
    get_prediction_history
)


# ==================================================
# 2. PATHS
# ==================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CNN_MODEL_PATH = os.path.join(
    RESULTS_DIR, "agrTrust_cnn_model.keras"
)

RF_MODEL_PATH = os.path.join(
    MODEL_DIR, "random_forest_model.joblib"
)

CLASS_NAMES_PATH = os.path.join(
    RESULTS_DIR, "class_names.json"
)

CONFIDENCE_THRESHOLD = 0.335

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ==================================================
# 3. FLASK APPLICATION
# ==================================================

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ==================================================
# 4. LOAD AI MODELS
# ==================================================

print("\n====================================")
print("LOADING AGRITRUST AI MODELS")
print("====================================")

cnn_model = tf.keras.models.load_model(
    CNN_MODEL_PATH,
    compile=False
)

cnn_model(
    tf.zeros((1, 224, 224, 3)),
    training=False
)

feature_extractor = tf.keras.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer("dense").output
)

rf_model = joblib.load(RF_MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

if len(class_names) != len(rf_model.classes_):
    raise ValueError(
        "Class names and Random Forest classes do not match."
    )

print("CNN model loaded successfully!")
print("Random Forest model loaded successfully!")
print("Total classes:", len(class_names))


# ==================================================
# 5. TOP-3 PREDICTIONS
# ==================================================

def get_top_predictions(probabilities):

    top_positions = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for position in top_positions:

        class_id = int(rf_model.classes_[position])

        class_name = class_names[class_id]

        crop_name, disease_name = class_name.split("___", 1)

        top_predictions.append({
            "crop": crop_name.replace("_", " "),
            "disease": disease_name.replace("_", " "),
            "confidence": round(
                float(probabilities[position]) * 100,
                2
            )
        })

    return top_predictions


# ==================================================
# 6. HOME PAGE
# ==================================================

@app.route("/")
def home():

    return send_file(
        os.path.join(FRONTEND_DIR, "index.html")
    )


# ==================================================
# 7. HEALTH CHECK
# ==================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "AgriTrust AI server is running",
        "total_classes": len(class_names),
        "feature_dimensions": 128
    })


# ==================================================
# 8. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_file):

    image = Image.open(image_file).convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==================================================
# 9. FILE VALIDATION
# ==================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==================================================
# 10. DISEASE PREDICTION
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error": "Only JPG, JPEG and PNG images are allowed."
            }), 400

        # IMAGE PREPROCESSING

        processed_image = preprocess_image(file)

        # CNN FEATURE EXTRACTION

        features = feature_extractor.predict(
            processed_image,
            verbose=0
        )

        if features.shape[1] != 128:

            raise ValueError(
                f"Expected 128 features, received {features.shape[1]}"
            )

        # RANDOM FOREST PREDICTION

        probabilities = rf_model.predict_proba(features)[0]

        # TOP-3 PREDICTIONS

        top_predictions = get_top_predictions(probabilities)

        # BEST PREDICTION

        prediction_position = int(
            np.argmax(probabilities)
        )

        predicted_class = int(
            rf_model.classes_[prediction_position]
        )

        confidence = float(
            probabilities[prediction_position]
        )

        class_name = class_names[predicted_class]

        crop_name, disease_name = class_name.split("___", 1)

        crop_display = crop_name.replace("_", " ")

        disease_display = disease_name.replace("_", " ")

        confidence_percentage = round(
            confidence * 100,
            2
        )

        # RELIABILITY DETECTION

        if confidence >= CONFIDENCE_THRESHOLD:

            reliability = "RELIABLE"

            message = (
                "The prediction passed the "
                "validation-selected confidence threshold."
            )

        else:

            reliability = "UNRELIABLE"

            message = (
                "The prediction has low confidence. "
                "Please upload a clearer image or consult "
                "an agricultural expert."
            )

        # SAVE PREDICTION

        save_prediction(
            crop_display,
            disease_display,
            confidence_percentage,
            reliability
        )

        # RESPONSE

        return jsonify({

            "success": True,

            "crop": crop_display,

            "disease": disease_display,

            "confidence": confidence_percentage,

            "confidence_threshold": round(
                CONFIDENCE_THRESHOLD * 100,
                2
            ),

            "reliability": reliability,

            "message": message,

            "top_predictions": top_predictions,

            "disclaimer": (
                "AI prediction only. Please consult an "
                "agricultural expert before making "
                "crop-treatment decisions."
            )

        })

    except Exception as error:

        print("\nPREDICTION ERROR:")

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ==================================================
# 11. PREDICTION HISTORY API
# ==================================================

@app.route("/history", methods=["GET"])
def prediction_history():

    try:

        records = get_prediction_history()

        return jsonify({

            "success": True,

            "total_records": len(records),

            "history": records

        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ==================================================
# 12. FILE SIZE ERROR
# ==================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": "Image size must be below 10 MB."

    }), 413


# ==================================================
# 13. START SERVER
# ==================================================

if __name__ == "__main__":

    print("\n====================================")
    print("AGRITRUST AI SERVER STARTING")
    print("====================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )