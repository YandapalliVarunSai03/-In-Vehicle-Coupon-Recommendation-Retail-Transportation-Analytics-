# In-Vehicle Coupon Recommendation Prediction

## Overview

This project aims to predict the likelihood of in-vehicle coupon usage based on factors like destination, passenger type, time, and more within the retail and transportation domain. **Random Forest** and **XGBoost** classifiers were used to predict coupon recommendation, achieving **76%** and **75%** accuracy respectively. Feature selection was optimized using **GridSearchCV** and **RandomizedSearchCV**, providing valuable insights for targeted marketing campaigns.

### Key Features:
- **Classification Task**: Predicting coupon recommendation based on customer features.
- **Modeling**: Utilized **Random Forest** and **XGBoost** for prediction.
- **Data Preprocessing**: Followed **Knowledge Discovery in Databases (KDD)** methodology.
- **Performance**: Achieved **76%** accuracy with Random Forest and **75%** accuracy with XGBoost.

## Table of Contents
1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Selection and Transformation](#feature-selection-and-transformation)
4. [Modeling and Performance](#modeling-and-performance)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation Metrics](#evaluation-metrics)

## Installation

### Required Libraries:
To install the necessary libraries, use the following command:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

## Data Preprocessing

The data was processed using the **Knowledge Discovery in Databases (KDD)** methodology, which includes the following key steps:

### 1. Handling Missing Data:
- **Imputation**: Missing values in categorical features (e.g., Bar, CoffeeHouse, CarryAway, RestaurantLessThan20) were imputed with the most frequent value (mode).
- **Column Removal**: The "car" feature was removed due to a high percentage of missing data and its irrelevance to the analysis.

### 2. Encoding and Transformation:
- **One-Hot Encoding**: Categorical variables (e.g., destination, passenger type, coupon) were encoded using one-hot encoding to ensure correct treatment in machine learning models.
- **Label Encoding**: Ordinal variables such as age and temperature were encoded using label encoding.
- **Frequency Encoding**: Features like Bar and CoffeeHouse were encoded using frequency-based encoding to capture customer behavior.

### 3. Feature Scaling:
- **Standardization**: Continuous variables (e.g., temperature, age) were scaled using the `StandardScaler` to have a mean of 0 and a standard deviation of 1.
- **Normalization**: Frequency-based features were normalized using `MinMaxScaler` to ensure that they contribute equally to the analysis.

### 4. Feature Selection:
A correlation analysis was performed to prioritize important features, ensuring only the most relevant predictors remained. This step improved model efficiency and accuracy.

### 5. Data Splitting:
The dataset was split into an **80% training set** and **20% testing set**, allowing the model to be evaluated on unseen data.

## Modeling and Performance

### Models Used:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost**

### Performance Metrics:
The following metrics were used to evaluate the models:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Percentage of correct positive predictions.
- **Recall**: Percentage of actual positive instances correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualizes the model's true positives, false positives, true negatives, and false negatives.

### Results:

| Metric                | Logistic Regression | Random Forest | XGBoost |
|-----------------------|---------------------|---------------|---------|
| **Accuracy**          | 68.31%              | 74.81%        | 75.60%  |
| **Precision**         | 0.62                | 0.69          | 0.70    |
| **Recall**            | 0.62                | 0.74          | 0.75    |
| **F1-Score**          | 0.69                | 0.72          | 0.72    |

- **Logistic Regression**: Accuracy of **68.31%** with lower precision (0.62) and recall (0.77), suggesting it missed some positive instances.
- **Random Forest**: Accuracy of **74.81%** with better precision (0.69) and recall (0.74) compared to Logistic Regression.
- **XGBoost**: Achieved the highest accuracy (**75.60%**) and demonstrated the best balance between precision (0.70) and recall (0.75).

## Hyperparameter Tuning

### Random Forest:
- Hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split` were tuned using **GridSearchCV**.
- Best parameters: `max_depth=30`, `min_samples_split=5`, `n_estimators=200`.
- Tuned accuracy: **73.98%**.

### XGBoost:
- Hyperparameter tuning focused on `learning_rate`, `max_depth`, `n_estimators`, `subsample`, and `colsample_bytree` using **GridSearchCV**.
- Best parameters: `learning_rate=0.1`, `max_depth=10`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8`.
- Tuned accuracy: **76.51%**.

### Impact of Hyperparameters:
- Increasing `max_depth` in tree-based models like Random Forest and XGBoost captures more complex relationships but increases the risk of overfitting.
- A smaller `learning_rate` in XGBoost improves generalization but requires more boosting rounds.
- `Subsample` and `colsample_bytree` values in XGBoost reduce overfitting by introducing randomness.

## Evaluation Metrics

- **Confusion Matrices** are included for each model to illustrate the true positives, false positives, true negatives, and false negatives.
- **XGBoost** consistently outperforms other models, showing the best ability to capture true positives and minimize false positives and false negatives.
