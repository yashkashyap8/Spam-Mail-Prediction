# SPAM Mail Prediction Model

## Overview
This project demonstrates a machine learning-based approach to predict whether a given email is spam or not. The model utilizes the Logistic Regression algorithm and is trained on textual data processed using TF-IDF vectorization. 

## Features
- **Machine Learning Model**: Logistic Regression
- **Text Processing**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Evaluation Metrics**: Accuracy Score

## Dataset
The project uses a dataset containing email text and labels indicating whether the email is spam (1) or not spam (0). The dataset should be loaded as a CSV file into a pandas DataFrame.

## Installation
To run the project, make sure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn
```

## Imports
The following libraries and modules are used in this project:

```python
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Model Workflow
1. **Data Preparation**: Load and preprocess the dataset (e.g., handle missing values, clean text data).
2. **Data Splitting**: Split the dataset into training and test sets using `train_test_split`.
3. **Text Vectorization**: Convert email text into numerical features using `TfidfVectorizer`.
4. **Model Training**: Train a Logistic Regression model on the training data.
5. **Evaluation**: Predict on the test data and compute accuracy.

## Results
The model demonstrates high accuracy on both training and test datasets, indicating its effectiveness in predicting spam emails. Below are the accuracy scores:

- **Training Accuracy**: 94.79%
- **Test Accuracy**: 93.96%



