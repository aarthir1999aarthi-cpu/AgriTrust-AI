# 🌱 AgriTrust AI
### Indian Crop Disease Prediction with AI Self-Reliability Detection

## 🚀 Live Application

**Try AgriTrust AI:** [Open Live Application](https://agritrust-ai-tymvzumvubabgzmbkr8sga.streamlit.app/)

AgriTrust AI is an AI-powered agriculture application that predicts plant diseases from leaf images and provides a reliability assessment of its predictions.

---

## 📌 Project Overview

AgriTrust AI is a Machine Learning-based agriculture project designed to help identify crop diseases using leaf images.

The application uses a CNN-based feature extraction model and a Random Forest classifier to predict crop disease categories. It also evaluates prediction confidence and displays whether the prediction meets the selected reliability threshold.

## 🎯 Problem Statement

Plant diseases can affect crop quality and agricultural productivity. Identifying diseases at an early stage can help farmers make informed decisions.

AgriTrust AI aims to support plant disease identification through image-based Artificial Intelligence and Machine Learning techniques.

## 🎯 Objectives

- To develop an AI-based crop disease prediction application.
- To identify crop and disease categories from leaf images.
- To apply CNN-based feature extraction and Machine Learning classification.
- To display prediction confidence and reliability status.
- To provide an interactive and user-friendly application.

## ✨ Key Features

- **Leaf Image Input:** Upload a leaf image or capture one using the camera.
- **Crop Identification:** Identify the predicted crop category.
- **Disease Prediction:** Predict the corresponding disease or healthy category.
- **Top Predictions:** Display the top three prediction results.
- **Confidence Score:** Show the model's prediction confidence.
- **AI Self-Reliability Detection:** Classify predictions based on a confidence threshold.
- **Feature Importance:** Display Random Forest feature importance.
- **Model Comparison:** Present model performance comparison.
- **Prediction History:** View previous prediction records.
- **CSV Download:** Download prediction information.

*Note: Reliability status is based on the configured confidence threshold. It does not guarantee that a prediction is correct.*

## 🧠 Machine Learning Models

The project uses:

1. **Convolutional Neural Network (CNN):** Extracts image features from leaf images.
2. **Random Forest Classifier:** Uses extracted features to classify crop disease categories.
3. **Decision Tree:** Included as an additional model in the project repository for model experimentation and comparison.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Pillow
- Streamlit
- Joblib
- SQLite
- Matplotlib

## 📂 Project Structure

```text
AgriTrust-AI/
│
├── models/
│   └── random_forest_model.joblib
│
├── results/
│   ├── agrTrust_cnn_model.keras
│   └── class_names.json
│
├── streamlit_app.py
├── app.py
├── database.py
├── requirements.txt
├── image_preprocessing.py
├── cnn_model.py
├── random_forest_model.py
├── decision_tree_model.py
├── reliability_detection.py
├── model_comparison.py
├── shap_explainability.py
└── README.md
```

*The repository also contains model evaluation reports, graphs, and supporting scripts.*

## 📊 Dataset Information

**Dataset Name:** PlantVillage

**Dataset Source:** [PlantVillage Dataset – GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

**Dataset Type:** Plant Leaf Images

**Dataset Size:** 54,305 images in the downloaded dataset.

### Dataset Usage

- Image preprocessing
- CNN feature extraction and training
- Plant disease classification
- Model evaluation

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aarthir1999aarthi-cpu/AgriTrust-AI.git
```

### 2. Open the Project Folder

```bash
cd AgriTrust-AI
```

### 3. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

## ▶️ How to Run

Start the Streamlit application:

```bash
python -m streamlit run streamlit_app.py
```

The application will open in your browser.

## 📈 Model Evaluation

The repository includes model evaluation reports and visualizations, such as:

- CNN training accuracy and loss graphs
- CNN confusion matrix
- CNN classification report
- Random Forest results
- Decision Tree results
- Model performance comparison
- Reliability detection results

Refer to the corresponding files in the repository for detailed evaluation results.

## 📸 Application Screenshots

Screenshots of the deployed application can be added here.

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

**Live URL:** https://agritrust-ai-tymvzumvubabgzmbkr8sga.streamlit.app/

## 📌 Conclusion

AgriTrust AI demonstrates how Artificial Intelligence and Machine Learning can support agriculture through image-based crop disease prediction.

By combining CNN-based feature extraction, Random Forest classification, and confidence-based reliability assessment, the project provides an interactive approach to plant disease identification.

## 👩‍💻 Project Repository

[View AgriTrust AI on GitHub](https://github.com/aarthir1999aarthi-cpu/AgriTrust-AI)
