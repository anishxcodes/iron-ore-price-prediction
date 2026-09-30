# Iron Ore AI

## Machine Learning-Based Prediction of Iron Ore Prices for Steel Industry Decision Support

Iron Ore AI is a Django-based machine learning application designed to predict iron ore prices and provide decision-support information for steel industry procurement planning.

The system uses historical iron ore and market-related data to train multiple machine learning regression models, compare their performance, select the best-performing model, and generate future price predictions.

---

## Project Objectives

- Predict iron ore prices using machine learning.
- Compare multiple regression models.
- Select the best-performing model using RMSE.
- Provide future price predictions based on market conditions.
- Explain model predictions using SHAP.
- Provide procurement decision-support signals.
- Maintain prediction history.
- Allow administrators to upload new datasets and retrain models.
- Provide dashboard visualization and CSV/PDF export.

---

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Backend
- Python
- Django

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- Joblib
- SHAP

### Database
- SQLite

### Reporting
- ReportLab

---

## Machine Learning Models

The application trains and compares:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

The model with the lowest RMSE is selected as the best-performing model.

### Evaluation Metrics

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² — R-squared

---

## Feature Engineering

The training pipeline creates the following historical features:

- Lag_1
- Lag_7
- MA_7
- MA_30

The moving-average features are calculated using previous observations to avoid using future price information during prediction.

---

## Input Features

The application uses:

- Steel Production Index
- Demand Index
- Supply Index
- Freight Cost (USD/tonne)
- Oil Price (USD/barrel)
- USD/CNY Exchange Rate
- Lag_1
- Lag_7
- MA_7
- MA_30

---

## Decision Support

The system calculates the expected percentage change between the current and predicted iron ore price.

The application uses the following decision-support thresholds:

- Predicted increase >= 5% → BUY EARLY
- Predicted decrease <= -5% → WAIT
- Otherwise → NORMAL PROCUREMENT

These are application-defined decision-support rules and are not financial advice.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain the contribution of input features to an individual prediction.

The application displays:

- Feature name
- SHAP contribution
- Impact direction
- SHAP visualization

SHAP values explain model behavior and should not be interpreted as proof of causal relationships.

---

## Application Workflow

Historical Dataset

↓

Data Validation

↓

Feature Engineering

↓

Train Multiple ML Models

↓

Evaluate Models

↓

Select Best Model

↓

Save Trained Model

↓

Future Prediction

↓

SHAP Explanation

↓

Decision Support

↓

Prediction History

---

## Main Features

### 1. Future Price Prediction

Users can enter market conditions and generate an estimated future iron ore price.

### 2. Dashboard

The dashboard provides:

- Model comparison
- RMSE comparison
- R² comparison
- Actual vs predicted values
- Feature importance

### 3. Prediction History

Previous predictions are stored in the database and can be reviewed later.

### 4. CSV/PDF Export

Prediction history can be exported as:

- CSV
- PDF

### 5. Dataset Upload

Administrators can upload a new dataset and retrain the machine learning models.

### 6. Role-Based Access

The application supports:

- Admin
- Analyst

Administrators can access dataset management and retraining features.

---

## Project Structure

```text
iron_ore_ai/
│
├── dataset/
│   └── iron_ore_price_dataset_without_lag.csv
│
├── iron_ore_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── ml_models/
│   └── iron_ore_model.pkl
│
├── prediction/
│   ├── admin.py
│   ├── apps.py
│   ├── context_processors.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── registration/
│   │   └── login.html
│   └── prediction/
│       ├── index.html
│       ├── future_prediction.html
│       ├── dashboard.html
│       ├── history.html
│       └── upload.html
│
├── manage.py
├── train_model.py
├── requirements.txt
└── README.md