# Credit Card Fraud Detection using Logistic Regression

## Overview

This project focuses on detecting fraudulent credit card transactions using a machine learning model. The dataset is highly imbalanced, with most transactions being legitimate and only a small fraction being fraudulent. To address this, undersampling was performed, and a Logistic Regression model was built to classify the transactions.

## Dataset

The dataset used for this project is `creditcard.csv`, which contains transactions made by credit cards in September 2013. It has the following attributes:

- **Time** : The seconds elapsed between this transaction and the first transaction in the dataset.
- **V1 to V28** : The result of a PCA transformation. The values are not interpretable individually.
- **Amount* *: The transaction amount.
- **Class** : The class label (`0` for legitimate transactions and `1` for fraudulent transactions).

## Project Workflow

1. **Data Loading & Exploration**:
   - Load the dataset and display its structure.
   - Check for missing values and basic statistics.
   - Analyze the distribution of legitimate vs fraudulent transactions.

2. **Under-Sampling**:
   - The dataset is highly imbalanced. Legitimate transactions far outnumber fraudulent ones.
   - A subset of the legitimate transactions is sampled to balance the dataset.

3. **Data Preprocessing**:
   - Split the dataset into features (X) and the target variable (Y).
   - Split the data into training and testing sets (80% training, 20% testing) using stratified sampling to maintain class balance in both sets.

4. **Model Training**:
   - Train a Logistic Regression model on the balanced training dataset.

5. **Model Evaluation**:
   - Evaluate the model using accuracy scores on both the training and testing sets.

## Dependencies

Make sure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
## Results
The Logistic Regression model achieves a high accuracy of 95.17% on the training data and 92.89% on the test data. These results suggest the model is effective in detecting fraudulent transactions, even with a highly imbalanced dataset.


