Dataset Preparation and Exploration

The selected dataset is a diabetes prediction dataset containing approximately 100,000 patient records and 15 predictor variables. The target variable, diabetes, represents a binary classification problem where 0 indicates the absence of diabetes and 1 indicates the presence of diabetes.

The dataset contains a mixture of numerical and categorical features. The categorical variables include gender, location, and smoking history, while the remaining variables are numerical clinical and demographic attributes. The dataset was inspected for missing values and duplicate records before model development.

The target variable is imbalanced, with a larger proportion of non-diabetic cases than diabetic cases. Therefore, in addition to accuracy, evaluation metrics such as AUC, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC) are used to provide a more comprehensive assessment of model performance.

The dataset was divided into training and testing subsets using an 80:20 split with stratification to preserve the class distribution in both subsets. The held-out test dataset was saved as test_data.csv for use in the Streamlit application.

## BEST MODEL
Decision Tree: The Decision Tree classifier achieved an accuracy of 97.14% and an AUC of 97.17%, indicating strong overall classification performance and excellent class discrimination. Its precision of 95.85% indicates that most positive diabetes predictions were correct. However, the recall of 69.29% indicates that the model failed to identify a portion of the actual diabetic cases. The F1 score of 80.44% and MCC of 0.8015 indicate a strong overall predictive relationship, although improving recall could further improve the model.

