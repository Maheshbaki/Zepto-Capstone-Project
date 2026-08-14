# Module 2 — EDA and Modeling

## Overview

This module performs exploratory data analysis, data cleaning, classification modeling, and fare regression using the Titanic dataset.

The workflow is:

Raw Titanic dataset
→ EDA and cleaning
→ Cleaned dataset
→ Train/test split
→ Preprocessing
→ Classification models
→ Model comparison and tuning
→ Fare regression
→ Final model persistence

---

## Files

- module_2_EDA.ipynb — exploratory data analysis and data cleaning
- module_2_modeling.ipynb — classification and regression modeling
- titanic.csv — original Titanic dataset
- titanic_cleaned.csv — cleaned dataset produced during EDA
- final_logistic_model.joblib — saved final classification pipeline

---

## 1. Exploratory Data Analysis and Cleaning

The original dataset contained:

- 891 rows
- 15 columns

Missing-value analysis identified missing values in:

- age
- embarked
- deck
- embark_town

The cleaned dataset contains:

- 889 rows
- 15 columns
- No missing values

The final cleaned columns were:

- survived
- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked
- class
- who
- adult_male
- embark_town
- alive
- alone
- age_group

EDA included:

- Missing-value analysis
- Mean, median, and mode analysis for fare
- Outlier analysis for fare and age
- Survival rate by sex
- Survival rate by passenger class
- Survival rate by sex and passenger class
- Correlation analysis
- Data visualization
- Standardization analysis

---

## 2. Modeling Preparation

The target variable for classification is:

survived

The following columns were excluded:

- survived — target variable
- alive — directly represents survival and would cause target leakage
- class — duplicate representation of pclass

### Numerical features

- pclass
- age
- sibsp
- parch
- fare

### Categorical features

- sex
- embarked
- who
- adult_male
- embark_town
- alone
- age_group

An 80/20 train/test split was used with:

- random_state = 42
- stratify = y

The target distribution was approximately:

- Not survived: 61.6%
- Survived: 38.4%

The training and testing target distributions remained similar after stratification.

---

## 3. Preprocessing

A ColumnTransformer and Pipeline were used for preprocessing.

### Numerical preprocessing

- Median imputation
- StandardScaler

### Categorical preprocessing

- Most-frequent imputation
- One-hot encoding
- handle_unknown = ignore

The preprocessing was fitted as part of each model pipeline to avoid data leakage between training and testing data.

---

# 4. Classification Models

The following classification approaches were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Random Forest with SMOTE
5. Tuned Random Forest using GridSearchCV

## Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8258 | 0.8136 | 0.7059 | 0.7559 | 0.8647 |
| Decision Tree | 0.7921 | 0.8163 | 0.5882 | 0.6838 | 0.8285 |
| Random Forest | 0.7978 | 0.7581 | 0.6912 | 0.7231 | 0.8235 |
| Random Forest + SMOTE | 0.8034 | 0.7463 | 0.7353 | 0.7407 | 0.8314 |
| Tuned Random Forest | 0.8202 | 0.8103 | 0.6912 | 0.7460 | 0.8402 |

---

## 5. Logistic Regression

Logistic Regression achieved:

- Accuracy: 82.58%
- Precision: 81.36%
- Recall: 70.59%
- F1 Score: 75.59%
- ROC-AUC: 86.47%

Confusion Matrix:

- True Negative: 99
- False Positive: 11
- False Negative: 20
- True Positive: 48

Logistic Regression provided the strongest overall classification performance among the tested models.

---

## 6. Decision Tree

The Decision Tree achieved:

- Accuracy: 79.21%
- Precision: 81.63%
- Recall: 58.82%
- F1 Score: 68.38%
- ROC-AUC: 82.85%

Confusion Matrix:

- True Negative: 101
- False Positive: 9
- False Negative: 28
- True Positive: 40

The trained tree structure was visualized using plot_tree.

Compared with Logistic Regression, the Decision Tree had lower accuracy, recall, F1 score, and ROC-AUC, although its precision was slightly higher.

---

## 7. Random Forest

The initial Random Forest achieved:

- Accuracy: 79.78%
- Precision: 75.81%
- Recall: 69.12%
- F1 Score: 72.31%
- ROC-AUC: 82.35%

Confusion Matrix:

- True Negative: 95
- False Positive: 15
- False Negative: 21
- True Positive: 47

---

## 8. SMOTE

The target variable showed moderate class imbalance:

- Class 0: approximately 61.6%
- Class 1: approximately 38.4%

SMOTE was applied only to the training data using an imblearn pipeline.

### Random Forest + SMOTE

- Accuracy: 80.34%
- Precision: 74.63%
- Recall: 73.53%
- F1 Score: 74.07%
- ROC-AUC: 83.14%

Confusion Matrix:

- True Negative: 93
- False Positive: 17
- False Negative: 18
- True Positive: 50

SMOTE increased recall from 69.12% to 73.53% and also improved F1 and ROC-AUC compared with the original Random Forest.

Precision decreased slightly.

This demonstrates the trade-off between identifying more minority-class observations and maintaining precision.

---

## 9. Random Forest Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the Random Forest.

The optimization metric was F1 score.

### Best Parameters

- n_estimators = 200
- max_depth = 5
- min_samples_split = 2
- min_samples_leaf = 1

Best cross-validation F1:

0.7654

### Tuned Random Forest Test Results

- Accuracy: 82.02%
- Precision: 81.03%
- Recall: 69.12%
- F1 Score: 74.60%
- ROC-AUC: 84.02%

The tuned Random Forest improved substantially over the initial Random Forest but did not outperform Logistic Regression overall.

---

## 10. Random Forest OOB Score

The tuned Random Forest was evaluated using the out-of-bag score.

OOB Score:

0.8256

The OOB score provides an internal estimate of Random Forest generalization performance using observations that were not included in individual bootstrap samples.

---

# 11. Fare Regression

A Linear Regression model was used to predict passenger fare.

### Regression Results

| Metric | Value |
|---|---:|
| MAE | 18.5530 |
| RMSE | 41.5188 |
| R² | 0.3539 |
| Adjusted R² | 0.2376 |

The model explains approximately 35.4% of the variation in fare on the test set.

The RMSE is substantially higher than the MAE, suggesting that some observations have relatively large prediction errors.

---

## 12. Residual Analysis and Heteroscedasticity

A residual plot was created to evaluate the regression errors.

The residual plot showed that the spread of residuals increased at higher predicted fare values.

This widening pattern suggests heteroscedasticity, meaning that the variance of the regression errors is not constant across the range of predictions.

Several large residuals were also visible, indicating that some observations were difficult for the Linear Regression model to predict accurately.

---

# 13. Final Classification Model

Logistic Regression was selected as the final classification model based on the test-set results.

Final performance:

- Accuracy: 82.58%
- Precision: 81.36%
- Recall: 70.59%
- F1 Score: 75.59%
- ROC-AUC: 86.47%

Logistic Regression achieved the highest accuracy, F1 score, and ROC-AUC among the evaluated models.

Random Forest with SMOTE achieved the highest recall at 73.53%, but its overall F1 and ROC-AUC were lower.

Therefore, Logistic Regression provides the strongest overall balance of classification performance for this dataset.

---

# 14. Model Persistence

The complete Logistic Regression preprocessing and modeling pipeline was saved using joblib.

File:

final_logistic_model.joblib

The saved pipeline was successfully reloaded and tested.

The reloaded model achieved:

Accuracy: 0.8258

This matched the original model accuracy, confirming that the complete preprocessing and model pipeline can be restored and used for predictions.

---

# 15. How to Run

Install the required libraries:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Run the notebooks in this order:

1. module_2_EDA.ipynb
2. module_2_modeling.ipynb

The EDA notebook produces:

titanic_cleaned.csv

The modeling notebook uses the cleaned dataset to train and evaluate the classification and regression models.

---

## Conclusion

This module demonstrates an end-to-end analytics and machine-learning workflow:

- Exploratory data analysis
- Data cleaning
- Feature preparation
- Train/test splitting
- Preprocessing pipelines
- Classification modeling
- Model evaluation
- Class imbalance analysis
- SMOTE
- Hyperparameter tuning with GridSearchCV
- Random Forest OOB evaluation
- Fare regression
- Residual analysis
- Model persistence using joblib

For Titanic survival classification, Logistic Regression provided the strongest overall performance in this experiment, while Random Forest with SMOTE provided the highest survivor recall.
