
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Classification",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🩺 Diabetes Classification Model Analyzer")

st.write(
    """
    This application demonstrates and compares six machine
    learning classification models for diabetes prediction.

    Upload the test dataset, select a machine learning model,
    and view its performance metrics, confusion matrix,
    classification report, and predictions.
    """
)


# =========================================================
# MODEL LOCATIONS
# =========================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "SVM": "model/svm.pkl"
}


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if os.path.exists(model_path):

            loaded_models[model_name] = joblib.load(
                model_path
            )

    return loaded_models


models = load_models()


# =========================================================
# CHECK MODELS
# =========================================================

if len(models) == 0:

    st.error(
        "No trained model files were found."
    )

    st.info(
        "Please make sure the six .pkl files are inside "
        "the model/ folder."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Model Settings")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📁 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)


# =========================================================
# LOAD TEST DATA
# =========================================================

if uploaded_file is not None:

    test_data = pd.read_csv(
        uploaded_file
    )

    st.success(
        "Test dataset uploaded successfully."
    )

else:

    if os.path.exists("test_data.csv"):

        test_data = pd.read_csv(
            "test_data.csv"
        )

        st.info(
            "No file uploaded. Using the default "
            "test_data.csv."
        )

    else:

        st.warning(
            "Please upload a CSV file to continue."
        )

        st.stop()


# =========================================================
# DATASET INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Number of Records",
        test_data.shape[0]
    )

with col2:

    st.metric(
        "Number of Columns",
        test_data.shape[1]
    )

with col3:

    st.metric(
        "Selected Model",
        selected_model
    )


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("🔍 Preview Test Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


# =========================================================
# TARGET VALIDATION
# =========================================================

TARGET = "diabetes"

if TARGET not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a "
        "'diabetes' column."
    )

    st.stop()


# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

X_test = test_data.drop(
    columns=[TARGET]
)

y_test = test_data[TARGET]


# =========================================================
# LOAD SELECTED MODEL
# =========================================================

model = models[selected_model]


# =========================================================
# PREDICTION
# =========================================================

try:

    y_pred = model.predict(X_test)

except Exception as error:

    st.error(
        "Prediction failed. Make sure the uploaded CSV "
        "has the same feature columns used during training."
    )

    st.exception(error)

    st.stop()


# =========================================================
# PROBABILITY SCORES
# =========================================================

try:

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_score = model.decision_function(
            X_test
        )

except Exception:

    y_score = None


# =========================================================
# EVALUATION METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


if y_score is not None:

    auc = roc_auc_score(
        y_test,
        y_score
    )

else:

    auc = np.nan


# =========================================================
# PERFORMANCE SECTION
# =========================================================

st.markdown("---")

st.header(
    f"📊 Performance of {selected_model}"
)


# =========================================================
# METRIC CARDS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with c2:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with c3:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


c4, c5, c6 = st.columns(3)

with c4:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with c5:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with c6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.markdown("---")

st.header("🔢 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Non-Diabetic",
        "Diabetic"
    ],
    yticklabels=[
        "Non-Diabetic",
        "Diabetic"
    ],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.header("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Non-Diabetic",
        "Diabetic"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# =========================================================
# PREDICTION RESULTS
# =========================================================

st.header("🔮 Prediction Results")

prediction_results = test_data.copy()

prediction_results["Predicted Diabetes"] = y_pred

if y_score is not None:

    prediction_results[
        "Diabetes Probability"
    ] = y_score

st.dataframe(
    prediction_results.head(100),
    use_container_width=True
)


# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown("---")

st.header("🏆 Model Comparison")

comparison_file = "model_comparison.csv"

if os.path.exists(comparison_file):

    comparison = pd.read_csv(
        comparison_file
    )

    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )

    # Identify best F1 model
    if "F1 Score" in comparison.columns:

        best_model = comparison.loc[
            comparison["F1 Score"].idxmax(),
            "ML Model"
        ]

        st.success(
            f"🏆 Best model based on F1 Score: "
            f"{best_model}"
        )

else:

    st.warning(
        "model_comparison.csv was not found."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "ML Assignment 2 | Diabetes Classification | "
    "Machine Learning Model Demonstration"
)