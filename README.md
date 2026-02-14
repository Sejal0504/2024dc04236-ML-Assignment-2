                                                   Assignment-2

# Assignment 2 — Machine Learning

**Name:** Sejal Patel  
**BITS ID:** 2024dc04236  
**Subject:** Machine Learning  
---

This project implements a complete end-to-end machine learning pipeline for income classification, including data preprocessing, model comparison, evaluation, and interactive deployment using Streamlit.

---

## Problem Statement

The goal of this project is to build, evaluate, and compare multiple classification models to predict whether an individual earns **more than $50K annually** using demographic and employment attributes.  

The system emphasizes reproducible preprocessing, model benchmarking, and deployment through an interactive Streamlit interface.

---

## Dataset Overview


## Dataset — Adult Income Classification
## Dataset Source: :- https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?utm_source=chatgpt.com&select=adult.csv

The Adult Income dataset is a binary classification dataset used to predict income class based on demographic and employment attributes.

After preprocessing:

- **Total records:** 48,842 rows  
- **Input features:** 14  
- **Target variable:** `income` (`<=50K`, `>50K`)

The features capture demographic, employment, and financial information relevant to income prediction.

---

## Feature Types & Data Description

The dataset contains a mix of **numerical** and **categorical** variables.

### Numerical Features

These represent measurable quantities:

- age → individual’s age  
- fnlwgt → census sample weight  
- capital-gain → capital profit earned  
- capital-loss → capital loss incurred  
- hours-per-week → weekly working hours  

These features were preserved as numeric values and scaled when required.

---

### Categorical Features

These describe qualitative attributes:

- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  

These variables were encoded using **One-Hot Encoding** to allow machine learning models to process them.

---

### Target Variable

- income  
  - <=50K → lower income class  
  - >50K  → higher income class  

Converted into binary format for classification modeling.

---

## Data Cleaning & Preprocessing Pipeline

To ensure reliable model performance, the dataset underwent the following preprocessing steps:

### Cleaning Steps

- Missing values marked with `?` were replaced and removed  
- Inconsistent labels and extra spaces were trimmed  
- Duplicate entries were checked and removed  

### Feature Preparation

- Categorical features encoded via One-Hot Encoding  
- Numerical features retained and scaled where necessary  
- Target variable converted into binary class  

### Train–Test Split

The dataset was divided into:

- **80% training data**  
- **20% testing data**

This prevents overfitting and ensures fair evaluation. A fixed random seed guarantees reproducibility.

---

## Data Pipeline Workflow

Raw Dataset  
→ Cleaning & Validation  
→ Encoding & Feature Preparation  
→ Train/Test Split  
→ Model Training  
→ Performance Evaluation  
→ Streamlit Deployment

---

### Machine Learning Workflow Summary

Dataset ingestion  
→ Data cleaning & validation  
→ Feature encoding  
→ Train–test split  
→ Model training  
→ Performance evaluation  
→ Interactive deployment

---

## Models Used

Six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy  
- ROC-AUC  
- Precision  
- Recall  
- F1-score  
- Matthews Correlation Coefficient (MCC)

---

## Model Performance Comparison


|     ML Model        | Accuracy |  AUC  | Precision | Recall |   F1  |  MCC  |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression |   0.846  | 0.905 |   0.732   | 0.598  | 0.658 | 0.565 |
| Decision Tree       |   0.806  | 0.747 |   0.604   | 0.631  | 0.617 | 0.488 |
| kNN                 |   0.827  | 0.865 |   0.672   | 0.588  | 0.627 | 0.517 |
| Naive Bayes         |   0.645  | 0.851 |   0.405   | 0.919  | 0.562 | 0.412 |
| Random Forest       |   0.846  | 0.901 |   0.722   | 0.614  | 0.663 | 0.567 |
| XGBoost             |   0.868  | 0.927 |   0.772   | 0.666  | 0.715 | 0.633 |


Note:-Model performance details as per train_models.py with a fixed train–test split

---

## Model Performance Observations


|     ML Model        |                     Performance                        |
|---------------------|--------------------------------------------------------|
| Logistic Regression |  Balanced baseline classifier with strong AUC          |
| Decision Tree       |  Interpretable but slightly lower generalization       |
| kNN                 |  Moderate performance; sensitive to scaling            |
| Naive Bayes         |  High recall but lower precision                       |
| Random Forest       |  Stable ensemble performance                           |
| XGBoost             |  Best overall model with highest predictive capability |

---

## Streamlit Deployment

An interactive Streamlit web application was developed to demonstrate real-time model evaluation.

Features include:

- CSV dataset upload  
- Model selection dropdown  
- Automated retraining  
- Performance metric visualization  
- Confusion matrix & classification report  


---

## Conclusion

This assignment demonstrates the practical implementation of an end-to-end machine learning workflow — from data preparation to model comparison and deployment.  

Ensemble methods, particularly XGBoost, achieved the best predictive performance, highlighting the importance of advanced boosting techniques in classification tasks.  

The Streamlit deployment validates the usability of machine learning models in interactive environments.

---










