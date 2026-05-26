# Machine Learning Workshop: End-to-End Pipeline & Medical Classification

This directory contains the complete source code, notebook architecture, and documentation for a hands-on workshop focused on building, evaluating, and optimizing an end-to-end supervised machine learning pipeline.

## Overview

The workshop walks through a binary classification problem using the real-world Breast Cancer Wisconsin (Diagnostic) Dataset. The objective is to predict whether a tumor is malignant (0) or benign (1) based on 30 digitized cellular features. 

The curriculum adopts an engineering approach, emphasizing data robustness, proper preprocessing techniques to avoid data leakage, and selecting metrics aligned with domain-specific risks (e.g., prioritizing Recall over global Accuracy to minimize dangerous false negatives in a medical context).

## Core Pipeline Architecture

The implementation is broken down into the following operational stages:

1.  **Data Loading & Engineering:** Automated and manual data ingestion methods using pandas DataFrames and numpy arrays.
2.  **Exploratory Data Analysis (EDA):** Checking dataset dimensions, class distributions, auditing for missing values, and visualizing feature variances using seaborn and matplotlib.
3.  **Data Preprocessing:** Splitting the data using a stratified 80/20 train/test distribution. Implementing `StandardScaler` to normalize features on the training split to avoid feature dominance.
4.  **Baseline & Metrics Definition:** Constructing a baseline `DecisionTreeClassifier`. Computing and explaining key evaluation metrics (Accuracy, Precision, Recall, and Multi-class Macro/Micro averages) alongside Confusion Matrices and ROC curves.
5.  **Demonstrating Preprocessing Importance:** A controlled experiment using `KNeighborsClassifier` trained on both unscaled and scaled data to visually prove the impact of normalization on distance-based models.
6.  **Model Comparison & Global Optimization:** Building modular `Pipeline` structures and separate hyperparameter grids for multiple model families (Decision Tree, KNN, Random Forest, and Logistic Regression). Running an automated `GridSearchCV` loop to identify the optimal configuration based on a custom Recall scorer.
7.  **Production Inference:** Simulating real-time inference on a raw, unscaled patient record using the optimized champion pipeline.

## Technical Stack

* **Data Manipulation:** pandas, numpy
* **Machine Learning:** scikit-learn (Pipelines, Model Selection, Metrics)
* **Visualization:** matplotlib, seaborn

## Setup and Execution

1.  Open the notebook file in Google Colab or your local Jupyter environment.
2.  Ensure your environment has the required dependencies installed:
    ```bash
    pip install pandas numpy notebook matplotlib seaborn scikit-learn
    ```
3.  Execute the cells chronologically from top to bottom. The code is seeded with a fixed random state to ensure exact repeatability of metrics and results.
