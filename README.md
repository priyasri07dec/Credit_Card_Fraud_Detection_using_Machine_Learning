# Credit_Card_Fraud_Detection_using_Machine_Learning

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where the number of legitimate transactions is significantly higher than fraudulent transactions.

This project develops an end-to-end Machine Learning solution for detecting fraudulent credit card transactions using the Credit Card Fraud Detection dataset from Kaggle.

The project covers the complete machine learning workflow, including:

* Data loading and understanding
* Data cleaning
* Duplicate removal
* Exploratory Data Analysis (EDA)
* Class imbalance analysis
* Feature distribution analysis
* Correlation analysis
* Outlier analysis
* Train-test split
* Feature scaling
* Baseline model comparison
* Class imbalance handling
* XGBoost model development
* Hyperparameter tuning
* Classification threshold optimization
* Final model evaluation
* Feature importance analysis
* Model serialization
* Streamlit deployment
* Single transaction prediction
* Batch transaction prediction

## Objectives

The main objectives of this project are:

* Analyze the credit card transaction dataset.
* Understand the characteristics of legitimate and fraudulent transactions.
* Identify and remove duplicate records.
* Perform exploratory data analysis to identify important patterns.
* Analyze the severe class imbalance in the target variable.
* Prepare the data for machine learning.
* Compare multiple classification algorithms.
* Handle class imbalance using appropriate techniques.
* Build and optimize an XGBoost classification model.
* Optimize the classification threshold using validation data.
* Evaluate the final model using fraud-focused performance metrics.
* Identify important features contributing to model predictions.
* Save the trained model and supporting artifacts.
* Develop a Streamlit application for real-time and batch fraud prediction.

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

The dataset contains transactions made by European cardholders over a period of two days.

### Dataset Characteristics

| Property                   |   Value |
| -------------------------- | ------: |
| Original Rows              | 284,807 |
| Original Columns           |      31 |
| Rows Removed as Duplicates |   1,081 |
| Final Rows                 | 283,726 |
| Features                   |      30 |
| Target Variable            | `Class` |
| Legitimate Transactions    | 283,253 |
| Fraudulent Transactions    |     473 |
| Legitimate Transactions    |  99.83% |
| Fraudulent Transactions    |   0.17% |

### Features

The dataset contains the following variables:

Time
V1 to V28
Amount
Class

The Class variable is the target variable.

### Target Encoding

| Class | Meaning                |
| ----: | ---------------------- |
|     0 | Legitimate Transaction |
|     1 | Fraudulent Transaction |

### Important Note About V1–V28

The variables V1 through V28 are anonymized PCA-transformed features.

Because these variables are anonymized, they do not have directly interpretable business meanings.

Therefore, an important feature such as V14 should not be interpreted as representing a specific customer characteristic, transaction type, or financial attribute.

The model uses statistical patterns within these anonymized features to distinguish fraudulent transactions from legitimate transactions.

## Project Overflow

Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Class Imbalance Analysis
   ↓
Feature Analysis
   ↓
Outlier Analysis
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Baseline Model Comparison
   ↓
Class Imbalance Handling
   ↓
Weighted XGBoost
   ↓
Hyperparameter Tuning
   ↓
Threshold Optimization
   ↓
Final Model Evaluation
   ↓
Feature Importance
   ↓
Model Serialization
   ↓
Streamlit Deployment

## 1. Data Loading and Understanding

The dataset was loaded using Pandas and examined to understand:

* Number of rows and columns
* Column names
* Data types
* Missing values
* Duplicate records
* Target distribution
* Basic statistical characteristics

The original dataset contained:

284,807 rows and 31 columns.

The dataset contains 30 predictor variables and one target variable.

## 2. Data Cleaning

Data quality checks were performed before model development.

The following checks were carried out:

Missing-value analysis
Duplicate-record analysis
Data-type verification
Statistical summary
Target-value verification
### Missing Values

No missing values were identified in the dataset.

Therefore, no missing-value imputation was required.

### Duplicate Removal

Exact duplicate rows were identified and removed.

A total of:

1,081 duplicate rows

were removed.

After removing duplicates, the dataset contained:

283,726 rows and 31 columns.

A second duplicate check confirmed that no exact duplicate rows remained.

## 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure of the data and identify patterns that could be useful for fraud detection.

The EDA included:

* Target class distribution
* Transaction amount analysis
* Time distribution analysis
* Class-wise transaction analysis
* Feature distribution analysis
* Correlation analysis
* Outlier analysis

## Class Distribution Analysis

The target variable was highly imbalanced.

After duplicate removal, the class distribution was:

|          Class |   Count | Percentage |
| -------------: | ------: | ---------: |
| 0 - Legitimate | 283,253 |     99.83% |
|      1 - Fraud |     473 |      0.17% |

This means that fraudulent transactions represent only a very small proportion of all transactions.

The severe imbalance makes fraud detection challenging.

A model that predicts almost every transaction as legitimate can still achieve very high accuracy while failing to identify a significant number of fraudulent transactions.

Therefore, accuracy alone was not used as the primary model-selection criterion.

## 5. Transaction Amount Analysis

The Amount variable was analyzed using descriptive statistics and visualizations.

Important statistics for the complete dataset were:

| Statistic          |    Amount |
| ------------------ | --------: |
| Mean               |     88.47 |
| Median             |     22.00 |
| Standard Deviation |    250.40 |
| Maximum            | 25,691.16 |

The large difference between the mean and median indicates that the Amount variable is strongly right-skewed.

Fraudulent transactions had:

| Statistic | Fraudulent Transactions |
| --------- | ----------------------: |
| Mean      |                  123.87 |
| Median    |                    9.82 |

Although the mean transaction amount for fraudulent transactions was higher, the median was lower.

Therefore, it would not be appropriate to conclude that fraudulent transactions are generally more expensive.

Log-scaled visualizations were used to improve the visibility of the distribution because of the strong skewness.

No log transformation was applied to the final model input solely based on this observation.

## 6. Time Analysis

The Time variable represents the elapsed time between transactions.

The distribution of Time was non-uniform.

The dataset showed different levels of transaction activity across different time periods, including a noticeable reduction in activity around approximately 90,000–115,000 seconds.

Class-wise analysis also showed some differences in the temporal distribution of legitimate and fraudulent transactions.

However, Time alone was not considered sufficient for fraud detection.

## 7. Feature Distribution Analysis

The distributions of the anonymized PCA features V1–V28 were analyzed.

Class-wise histograms and boxplots were used to compare legitimate and fraudulent transactions.

Several features showed noticeable distribution shifts between the two classes.

Features such as:

V14
V17
V12
V10
V16
V7

showed particularly noticeable differences between legitimate and fraudulent transactions.

However, substantial overlap between the two classes was also observed.

This indicates that fraud detection requires combining information from multiple features rather than relying on a single variable.

## 8. Correlation Analysis

Correlation analysis was performed to understand the linear relationship between the predictor variables and the target variable.

The features with the strongest absolute correlations with Class were:

| Rank | Feature | Correlation with Class |
| ---: | ------- | ---------------------: |
|    1 | V17     |                -0.3135 |
|    2 | V14     |                -0.2934 |
|    3 | V12     |                -0.2507 |
|    4 | V10     |                -0.2066 |
|    5 | V16     |                -0.1872 |
|    6 | V7      |                -0.1724 |
|    7 | V11     |                 0.1491 |
|    8 | V4      |                 0.1293 |
|    9 | V18     |                -0.1053 |
|   10 | V1      |                -0.0945 |

Time and Amount had relatively weak linear correlations with the target:

Correlation analysis was used as an exploratory technique.

It was not used as the sole basis for feature selection because non-linear machine learning models can identify useful patterns even when linear correlation is weak.

## 9. Outlier Analysis

The Interquartile Range (IQR) method was used to identify statistical outliers.

Several variables contained observations outside the IQR boundaries.

Examples included:

* V27
* Amount
* V28
* V20
* V8
* V6
* V23
* V12
* V21
* V14
* V2
* V5

The presence of statistical outliers does not necessarily indicate erroneous data.

In fraud detection, unusual observations may contain valuable information because fraudulent transactions can differ substantially from normal transaction patterns.

Therefore, the identified outliers were retained.

The main data-cleaning step was removal of exact duplicate records.

## 10. EDA Summary

The EDA showed the following major findings:

* The dataset is extremely imbalanced.
* Fraudulent transactions represent approximately 0.17% of the cleaned dataset.
* No missing values were present.
* 1,081 exact duplicate records were removed.
* Amount is strongly right-skewed.
* Time has a non-uniform distribution.
* Several anonymized PCA features show different distributions between legitimate and fraudulent transactions.
* V17, V14, V12, V10, V16 and V7 have relatively strong absolute correlations with the target.
* Several variables contain statistical outliers.
* Outliers were retained because unusual observations may be meaningful in fraud detection.

These findings indicated that:

* Class imbalance needs to be addressed.
* Accuracy alone is not sufficient for model evaluation.
* Appropriate evaluation metrics are required.
* Feature scaling is required for some algorithms.
* Non-linear models may be useful for identifying complex fraud patterns.

## 11. Train-Test Split

The target variable was separated from the input features:
X = df.drop(columns="Class")
y = df["Class"]

The data was split into training and testing datasets using stratified sampling:
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

The training and testing sets retained approximately the same class distribution as the original dataset.

