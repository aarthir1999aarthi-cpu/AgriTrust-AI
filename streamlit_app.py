import os
import json
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import streamlit as st

from PIL import Image
from database import save_prediction, get_prediction_history


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AgriTrust AI",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_MODEL_PATH = os.path.join(
    BASE_DIR, "results", "agrTrust_cnn_model.keras"
)

RF_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "random_forest_model.joblib"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "results", "class_names.json"
)

CONFIDENCE_THRESHOLD = 0.335


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5faf5;
}

.title {
    color: #176b35;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    color: #52745d;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD AI MODELS
# ==========================================

@st.cache_resource
def load_models():

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

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != len(rf_model.classes_):
        raise ValueError(
            "Class names and Random Forest classes do not match."
        )

    if feature_extractor.output_shape[-1] != 128:
        raise ValueError(
            "CNN feature dimensions must be 128."
        )

    return feature_extractor, rf_model, class_names


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

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


# ==========================================
# 5. TOP 3 PREDICTIONS
# ==========================================

def get_top_predictions(probabilities, rf_model, class_names):

    top_positions = np.argsort(probabilities)[::-1][:3]

    predictions = []

    for position in top_positions:

        class_id = int(rf_model.classes_[position])

        class_name = class_names[class_id]

        crop_name, disease_name = class_name.split("___", 1)

        predictions.append({
            "Crop": crop_name.replace("_", " "),
            "Disease": disease_name.replace("_", " "),
            "Confidence (%)": round(
                float(probabilities[position]) * 100, 2
            )
        })

    return predictions


# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    '<p class="title">🌱 AgriTrust AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Indian Crop Disease Prediction with AI Self-Reliability Detection'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 7. SIDEBAR
# ==========================================

st.sidebar.title("🌿 AgriTrust AI")

page = st.sidebar.radio(
    "Navigation",
    ["Disease Detection", "Prediction History", "About"]
)


# ==========================================
# 8. LOAD MODELS
# ==========================================

try:

    with st.spinner("Loading CNN and Random Forest models..."):

        feature_extractor, rf_model, class_names = load_models()

    st.sidebar.success("AI Models Loaded")

except Exception as error:

    st.error("Unable to load AI models.")

    st.exception(error)

    st.stop()


# ==========================================
# 9. DISEASE DETECTION
# ==========================================

if page == "Disease Detection":

    st.header("🔍 Crop Disease Detection")

    st.write(
        "Upload a leaf image to identify the crop "
        "and predict its possible disease."
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        if uploaded_file.size > 10 * 1024 * 1024:

            st.error("Image size must be below 10 MB.")

        else:

            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Uploaded Leaf Image")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Image Information")

                st.write("File name:", uploaded_file.name)

                st.write("Image size:", image.size)

                st.write("Image format:", image.format)

            if st.button(
                "🔬 Analyze Crop",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Analyzing image using AI models..."
                    ):

                        processed_image = preprocess_image(image)

                        # CNN FEATURE EXTRACTION

                        features = feature_extractor.predict(
                            processed_image,
                            verbose=0
                        )

                        if features.shape[1] != 128:

                            raise ValueError(
                                "Expected 128 CNN features."
                            )

                        # RANDOM FOREST PREDICTION

                        probabilities = rf_model.predict_proba(
                            features
                        )[0]

                        # TOP 3 PREDICTIONS

                        top_predictions = get_top_predictions(
                            probabilities,
                            rf_model,
                            class_names
                        )

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

                        crop_name, disease_name = class_name.split(
                            "___", 1
                        )

                        crop_display = crop_name.replace("_", " ")

                        disease_display = disease_name.replace("_", " ")

                        confidence_percentage = round(
                            confidence * 100, 2
                        )

                        # RELIABILITY DETECTION

                        if confidence >= CONFIDENCE_THRESHOLD:

                            reliability = "RELIABLE"

                        else:

                            reliability = "UNRELIABLE"

                        # SAVE PREDICTION

                        save_prediction(
                            crop_display,
                            disease_display,
                            confidence_percentage,
                            reliability
                        )

                    st.success("Analysis completed!")

                    st.divider()

                    st.subheader("🌾 Prediction Results")

                    result1, result2 = st.columns(2)

                    with result1:

                        st.metric(
                            "Identified Crop",
                            crop_display.title()
                        )

                        st.metric(
                            "Predicted Disease",
                            disease_display.title()
                        )

                    with result2:

                        st.metric(
                            "Confidence Score",
                            f"{confidence_percentage}%"
                        )

                        st.metric(
                            "Reliability Status",
                            reliability
                        )

                    if reliability == "RELIABLE":

                        st.success(
                            "The prediction passed the "
                            "validation-selected confidence threshold."
                        )

                    else:

                        st.warning(
                            "The prediction has low confidence. "
                            "Please upload a clearer image or consult "
                            "an agricultural expert."
                        )

                    st.write(
                        f"Confidence threshold: "
                        f"{CONFIDENCE_THRESHOLD * 100:.2f}%"
                    )

                    st.divider()

                    st.subheader("📊 Top 3 Predictions")

                    prediction_df = pd.DataFrame(top_predictions)

                    st.dataframe(
                        prediction_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.bar_chart(
                        prediction_df.set_index("Disease")[
                            "Confidence (%)"
                        ]
                    )

                    st.info(
                        "AI prediction only. Please consult an "
                        "agricultural expert before making "
                        "crop-treatment decisions."
                    )

                except Exception as error:

                    st.error("Prediction failed.")

                    st.exception(error)


# ==========================================
# 10. PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.header("📚 Prediction History")

    try:

        records = get_prediction_history()

        if records:

            if isinstance(records[0], dict):

                history_df = pd.DataFrame(records)

            else:

                history_df = pd.DataFrame(
                    records,
                    columns=[
                        "id",
                        "crop",
                        "disease",
                        "confidence",
                        "reliability",
                        "prediction_date"
                    ]
                )

            st.metric(
                "Total Predictions",
                len(history_df)
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No prediction history available yet.")

    except Exception as error:

        st.error("Unable to retrieve prediction history.")

        st.exception(error)


# ==========================================
# 11. ABOUT
# ==========================================

elif page == "About":

    st.header("🌱 About AgriTrust AI")

    st.write("""
    AgriTrust AI is an intelligent crop disease prediction system.

    **Main Technologies:**

    - Convolutional Neural Network (CNN)
    - Random Forest Classifier
    - Feature Extraction
    - Confidence Score Analysis
    - AI Self-Reliability Detection
    - SQLite Prediction History

    **Main Features:**

    1. Leaf image upload
    2. Crop identification
    3. Disease prediction
    4. Top-3 predictions
    5. Confidence score
    6. Reliable / Unreliable detection
    7. Prediction history
    """)

    st.info(
        "AgriTrust AI supports crop disease identification. "
        "It does not replace professional agricultural advice."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "AgriTrust AI | AI-Based Indian Crop Disease Prediction"
)