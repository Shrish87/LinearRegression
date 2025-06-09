# 🏡 Housing Price Prediction using Linear Regression

This repository contains a Python script that performs **Linear Regression** on a housing dataset (`Housing.csv`). The model is trained to predict **house prices** based on the **area of the property**, using common machine learning steps such as data preprocessing, model training, evaluation, and visualization.

---

## 📁 Dataset

- **File used**: `Housing.csv`
- This file should contain at least the following numeric columns:
  - `area`: Size of the house (independent variable)
  - `price`: Price of the house (target variable)
- You may add or modify columns as needed for experimentation.

---

## 🔧 Features and Workflow

### 1. Import and preprocess the dataset
- Load the CSV using `pandas`
- Handle missing values
- Select numeric columns
- Choose `area` as the feature and `price` as the target

### 2. Split data
- Uses `train_test_split` to split the dataset (80% train, 20% test)

### 3. Train the model
- Linear Regression is applied using `scikit-learn`

### 4. Evaluate model performance
- Metrics printed:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R-squared (R²)**

### 5. Visualize results
- Plot of actual vs predicted prices
- Regression line over the test data

---

## 📊 Sample Output

- Printed metrics:
  ```text
  MAE: 500000.00
  MSE: 600000000.00
  R²: 0.75
