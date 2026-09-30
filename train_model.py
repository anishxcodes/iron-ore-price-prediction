import os

import joblib

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET_PATH = (
    "dataset/"
    "iron_ore_price_dataset_without_lag.csv"
)

MODEL_PATH = (
    "ml_models/"
    "iron_ore_model.pkl"
)


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def train_model():

    print("\nStarting model training...")


    # --------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------

    df = pd.read_csv(DATASET_PATH)


    # --------------------------------------------------
    # 2. SORT BY DATE
    # --------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


    df = df.sort_values(
        "Date"
    )


    # --------------------------------------------------
    # 3. CONVERT NUMERIC COLUMNS
    # --------------------------------------------------

    numeric_columns = [

        "Iron_Ore_Price_USD_per_tonne",

        "Steel_Production_Index",

        "Demand_Index",

        "Supply_Index",

        "Freight_Cost_USD_per_tonne",

        "Oil_Price_USD_per_barrel",

        "USD_CNY_Exchange_Rate",

    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------------
    # 4. FEATURE ENGINEERING
    # --------------------------------------------------

    price_column = (
        "Iron_Ore_Price_USD_per_tonne"
    )


    df["Lag_1"] = (
        df[price_column].shift(1)
    )


    df["Lag_7"] = (
        df[price_column].shift(7)
    )


    df["MA_7"] = (
        df[price_column]
        .shift(1)
        .rolling(7)
        .mean()
    )


    df["MA_30"] = (
        df[price_column]
        .shift(1)
        .rolling(30)
        .mean()
    )


    # --------------------------------------------------
    # 5. REMOVE MISSING VALUES
    # --------------------------------------------------

    df = df.dropna()


    if len(df) < 10:

        raise ValueError(
            "Not enough valid records for model training."
        )


    # --------------------------------------------------
    # 6. FEATURES
    # --------------------------------------------------

    feature_columns = [

        "Steel_Production_Index",

        "Demand_Index",

        "Supply_Index",

        "Freight_Cost_USD_per_tonne",

        "Oil_Price_USD_per_barrel",

        "USD_CNY_Exchange_Rate",

        "Lag_1",

        "Lag_7",

        "MA_7",

        "MA_30",

    ]


    X = df[feature_columns]

    y = df[price_column]


    # --------------------------------------------------
    # 7. CHRONOLOGICAL TRAIN-TEST SPLIT
    # --------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )


    X_train = X.iloc[
        :split_index
    ]


    X_test = X.iloc[
        split_index:
    ]


    y_train = y.iloc[
        :split_index
    ]


    y_test = y.iloc[
        split_index:
    ]


    # --------------------------------------------------
    # 8. CREATE MODELS
    # --------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

    }


    # --------------------------------------------------
    # 9. TRAIN AND EVALUATE
    # --------------------------------------------------

    results = {}

    predictions = {}


    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )


        model.fit(
            X_train,
            y_train
        )


        y_pred = model.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            y_pred
        )


        mse = mean_squared_error(
            y_test,
            y_pred
        )


        rmse = np.sqrt(mse)


        r2 = r2_score(
            y_test,
            y_pred
        )


        results[name] = {

            "MAE": float(mae),

            "MSE": float(mse),

            "RMSE": float(rmse),

            "R2": float(r2),

        }


        predictions[name] = y_pred


        print(
            f"MAE: {mae:.4f}"
        )

        print(
            f"MSE: {mse:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"R2: {r2:.4f}"
        )


    # --------------------------------------------------
    # 10. SELECT BEST MODEL
    # --------------------------------------------------

    best_model_name = min(

        results,

        key=lambda name:
            results[name]["RMSE"]

    )


    best_model = models[
        best_model_name
    ]


    print(
        f"\nBest Model: {best_model_name}"
    )


    # --------------------------------------------------
    # 11. FEATURE IMPORTANCE
    # --------------------------------------------------

    feature_importance = {}


    if hasattr(
        best_model,
        "feature_importances_"
    ):

        feature_importance = dict(
            zip(
                feature_columns,
                best_model.feature_importances_
            )
        )

    else:

        feature_importance = dict(
            zip(
                feature_columns,
                np.abs(
                    best_model.coef_
                )
            )
        )


    # --------------------------------------------------
    # 12. SAVE PREDICTIONS
    # --------------------------------------------------

    best_predictions = predictions[
        best_model_name
    ]


    prediction_records = []


    for actual, predicted in zip(
        y_test,
        best_predictions
    ):

        prediction_records.append({

            "actual":
                float(actual),

            "predicted":
                float(predicted),

        })


    # --------------------------------------------------
    # 13. SAVE MODEL
    # --------------------------------------------------

    os.makedirs(
        "ml_models",
        exist_ok=True
    )


    model_package = {

        "model":
            best_model,

        "model_name":
            best_model_name,

        "features":
            feature_columns,

        "results":
            results,

        "predictions":
            prediction_records,

        "feature_importance":
            feature_importance,

        "X_train":
            X_train,

    }


    joblib.dump(

        model_package,

        MODEL_PATH

    )


    print(
        "\nModel saved successfully."
    )


    # --------------------------------------------------
    # 14. RETURN TRAINING INFORMATION
    # --------------------------------------------------

    return {

        "best_model":
            best_model_name,

        "results":
            results,

        "records":
            len(df),

    }


# --------------------------------------------------
# RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    train_model()