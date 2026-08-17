# Diabetes Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting diabetes using demographic, lifestyle, and clinical features.

The project implements six classification algorithms and evaluates their performance using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application was also developed to allow users to upload test data, select a machine learning model and visualize the resulting predictions and evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is a diabetes prediction dataset containing 100,000 instances and 15 predictor features.

The target variable is:

`diabetes`

The target represents a binary classification problem:

- 0 – Non-Diabetic
- 1 – Diabetic

### Features

The dataset contains demographic, lifestyle and clinical features.

The categorical features include:

- gender
- location
- smoking_history

The dataset also contains numerical features related to demographic and clinical characteristics.

### Data Preprocessing

The dataset was divided into training and testing datasets using an 80:20 stratified split.

Categorical variables were encoded using One-Hot Encoding.

Numerical variables were standardized using StandardScaler where required.

The preprocessing steps were implemented using a scikit-learn ColumnTransformer and Pipeline so that the same preprocessing is applied during both training and prediction.

The held-out test dataset is provided as:

`test_data.csv`

---

## c. GitHub Repository Link

GitHub Repository:

PASTE YOUR GITHUB REPOSITORY LINK HERE

---

## d. Models Used

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| SVM | | | | | | |

---

# Model Performance Observations

## Logistic Regression

Logistic Regression was used as a baseline classification model. Its performance was evaluated using all six required metrics. The model provides a relatively simple and interpretable approach to the binary classification problem.

Based on the experimental results, its accuracy, AUC, precision, recall, F1 score and MCC were compared with the other models.

## Decision Tree

The Decision Tree model achieved an accuracy of 97.14%, AUC of 97.17%, precision of 95.85%, recall of 69.29%, F1 score of 80.44% and MCC of 0.8015.

The high accuracy and AUC indicate strong classification performance. However, the lower recall indicates that some actual diabetic cases were not identified by the model.

## KNN

KNN classifies observations based on their similarity to neighboring observations.

Feature scaling is particularly important for KNN because the distance calculation is affected by the scale of the input features.

Its performance was compared with the other classification algorithms using the same test dataset.

## Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on the assumption that features are conditionally independent given the target class.

The model provides a computationally efficient baseline and its performance was compared against the other models.

## Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees.

The ensemble approach can capture nonlinear relationships and generally provides more robust predictions than a single decision tree.

Its performance was evaluated using the same six metrics.

## Support Vector Machine

Support Vector Machine was used to identify a decision boundary between the two diabetes classes.

The model was evaluated using Accuracy, AUC, Precision, Recall, F1 Score and MCC.

---

# Overall Winner

The overall winner was selected primarily based on F1 Score while also considering AUC, MCC, Precision, Recall and Accuracy.

**Overall Winner: INSERT MODEL NAME**

The selected model achieved the strongest overall balance of classification performance according to the experimental results.

---

# Streamlit Application

An interactive Streamlit application was developed for demonstrating the trained models.

The application provides the following features:

### Dataset Upload

Users can upload the test dataset in CSV format.

### Model Selection

Users can select one of the six implemented machine learning models using a dropdown menu.

### Evaluation Metrics

The application displays:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- MCC Score

### Confusion Matrix

A confusion matrix is displayed for the selected model.

### Classification Report

A classification report containing precision, recall and F1 score for each class is displayed.

### Prediction Results

The application displays the actual and predicted diabetes values.

### Model Comparison

The application displays the performance of all six implemented models.

---

# Project Structure

```text
2025AC05139_ML_Assignment2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
└── notebooks/
    └── ML_Assignment_2.ipynb