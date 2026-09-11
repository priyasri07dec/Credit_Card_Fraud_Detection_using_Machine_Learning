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
Identify important features contributing to model predictions.

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

Target class distribution
Transaction amount analysis
Time distribution analysis
Class-wise transaction analysis
Feature distribution analysis
Correlation analysis
Outlier analysis

## Class Distribution Analysis

The target variable was highly imbalanced.

After duplicate removal, the class distribution was:

|          Class |   Count | Percentage |
| -------------: | ------: | ---------: |
| 0 - Legitimate | 283,253 |     99.83% |
|      1 - Fraud |     473 |      0.17% |




Save the trained model and supporting artifacts.
Develop a Streamlit application for real-time and batch fraud prediction.
