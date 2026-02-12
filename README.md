\# Adult Income Classification — ML Assignment 2



\## Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether an individual earns more than \\$50K annually based on demographic and employment attributes. The goal is to compare model performance using evaluation metrics and deploy an interactive Streamlit web application for real-time predictions.



---



\## Dataset Description

The Adult Income dataset contains demographic and employment-related features such as age, workclass, education, marital status, occupation, hours worked, and capital gains/losses.



Target variable:

\- income → <=50K or >50K



The dataset was cleaned by removing missing values, trimming inconsistent labels, and splitting into training and testing sets for fair evaluation.



---



\## Models Used \& Evaluation Metrics



| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |

|----------|----------|------|-----------|--------|------|------|

| Logistic Regression | 0.846 | 0.905 | 0.732 | 0.598 | 0.658 | 0.565 |

| Decision Tree | 0.806 | 0.747 | 0.604 | 0.631 | 0.617 | 0.488 |

| kNN | 0.827 | 0.865 | 0.672 | 0.588 | 0.627 | 0.517 |

| Naive Bayes | 0.645 | 0.851 | 0.405 | 0.919 | 0.562 | 0.412 |

| Random Forest | 0.846 | 0.901 | 0.722 | 0.614 | 0.663 | 0.567 |

| XGBoost | \*\*0.868\*\* | \*\*0.927\*\* | \*\*0.772\*\* | \*\*0.666\*\* | \*\*0.715\*\* | \*\*0.633\*\* |



---



\## Model Performance Observations



| Model | Observation |

|------|-------------|

| Logistic Regression | Balanced performance with strong AUC; good baseline classifier. |

| Decision Tree | Fast and interpretable but slightly lower generalization. |

| kNN | Moderate performance; sensitive to feature scaling. |

| Naive Bayes | High recall but low precision; tends to over-predict positive class. |

| Random Forest | Robust ensemble model with stable metrics. |

| XGBoost | Best overall performer; highest accuracy and MCC indicating strong predictive capability. |



---



\## Streamlit Application

The deployed Streamlit app allows:

\- Uploading CSV test data

\- Selecting classification models

\- Viewing evaluation metrics

\- Displaying confusion matrix and classification reports



This demonstrates end-to-end ML deployment workflow.



---



\## Author

Name:- Sejal Patel

Bits ID:- 2024dc04236

Subject:- Machine Learning



