import csv
import shutil
from pathlib import Path

import joblib
import pandas as pd

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
)

from .forms import FuturePredictionForm, CSVUploadForm
from .models import PredictionHistory


# ==================================================
# ADMIN ROLE CHECK
# ==================================================

def is_admin(user):

    return user.is_authenticated and (
        user.is_superuser
        or user.groups.filter(name="Admin").exists()
    )


# ==================================================
# HOME PAGE
# ==================================================

@login_required
def home(request):

    return render(
        request,
        "prediction/index.html"
    )


# ==================================================
# FUTURE PRICE PREDICTION
# ==================================================

@login_required
def future_prediction(request):

    model_path = (
        settings.BASE_DIR
        / "ml_models"
        / "iron_ore_model.pkl"
    )

    dataset_path = (
        settings.BASE_DIR
        / "dataset"
        / "iron_ore_price_dataset_without_lag.csv"
    )

    form = FuturePredictionForm()

    prediction = None
    current_price = None
    price_change = None
    recommendation = None
    explanation = None
    model_name = None
    shap_explanation = []

    if request.method == "POST":

        form = FuturePredictionForm(request.POST)

        if form.is_valid():

            try:

                # -----------------------------------------
                # LOAD TRAINED MODEL
                # -----------------------------------------

                package = joblib.load(model_path)

                model = package["model"]

                model_name = package["model_name"]

                features = package["features"]


                # -----------------------------------------
                # LOAD DATASET
                # -----------------------------------------

                data = pd.read_csv(dataset_path)

                data["Date"] = pd.to_datetime(
                    data["Date"],
                    dayfirst=True,
                    errors="coerce"
                )

                data = data.dropna(
                    subset=["Date"]
                )

                data = data.sort_values(
                    "Date"
                )


                # -----------------------------------------
                # CONVERT PRICE TO NUMERIC
                # -----------------------------------------

                data[
                    "Iron_Ore_Price_USD_per_tonne"
                ] = pd.to_numeric(
                    data[
                        "Iron_Ore_Price_USD_per_tonne"
                    ],
                    errors="coerce"
                )

                data = data.dropna(
                    subset=[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                )


                # -----------------------------------------
                # CREATE ENGINEERED FEATURES
                # -----------------------------------------

                data["Lag_1"] = (
                    data[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                    .shift(1)
                )

                data["Lag_7"] = (
                    data[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                    .shift(7)
                )

                data["MA_7"] = (
                    data[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                    .shift(1)
                    .rolling(7)
                    .mean()
                )

                data["MA_30"] = (
                    data[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                    .shift(1)
                    .rolling(30)
                    .mean()
                )

                data = data.dropna()


                # -----------------------------------------
                # LATEST MARKET INFORMATION
                # -----------------------------------------

                latest = data.iloc[-1]

                current_price = float(
                    latest[
                        "Iron_Ore_Price_USD_per_tonne"
                    ]
                )


                # -----------------------------------------
                # USER INPUTS
                # -----------------------------------------

                steel_production = (
                    form.cleaned_data[
                        "steel_production"
                    ]
                )

                demand = (
                    form.cleaned_data[
                        "demand"
                    ]
                )

                supply = (
                    form.cleaned_data[
                        "supply"
                    ]
                )

                freight = (
                    form.cleaned_data[
                        "freight"
                    ]
                )

                oil = (
                    form.cleaned_data[
                        "oil"
                    ]
                )

                usd_cny = (
                    form.cleaned_data[
                        "usd_cny"
                    ]
                )


                # -----------------------------------------
                # PREPARE PREDICTION DATA
                # -----------------------------------------

                input_data = pd.DataFrame([{

                    "Steel_Production_Index":
                        steel_production,

                    "Demand_Index":
                        demand,

                    "Supply_Index":
                        supply,

                    "Freight_Cost_USD_per_tonne":
                        freight,

                    "Oil_Price_USD_per_barrel":
                        oil,

                    "USD_CNY_Exchange_Rate":
                        usd_cny,

                    "Lag_1":
                        float(
                            latest["Lag_1"]
                        ),

                    "Lag_7":
                        float(
                            latest["Lag_7"]
                        ),

                    "MA_7":
                        float(
                            latest["MA_7"]
                        ),

                    "MA_30":
                        float(
                            latest["MA_30"]
                        ),

                }])


                # -----------------------------------------
                # ENSURE FEATURE ORDER
                # -----------------------------------------

                input_data = input_data[
                    features
                ]


                # -----------------------------------------
                # PREDICT PRICE
                # -----------------------------------------

                prediction = float(
                    model.predict(
                        input_data
                    )[0]
                )


                # -----------------------------------------
                # SHAP EXPLANATION
                # -----------------------------------------

                try:

                    X_train = package.get("X_train")

                    if model_name == "Linear Regression":

                        if X_train is not None:
                            explainer = shap.LinearExplainer(
                                model,
                                X_train
                            )
                        else:
                            explainer = shap.Explainer(
                                model,
                                input_data
                            )

                    else:

                        explainer = shap.TreeExplainer(
                            model
                        )

                    shap_values = explainer(
                        input_data
                    )

                    contributions = shap_values.values[0]

                    shap_explanation = []

                    for feature, contribution in zip(
                        features,
                        contributions
                    ):

                        shap_explanation.append({
                            "feature": feature,
                            "contribution": float(
                                contribution
                            ),
                            "impact": (
                                "Positive"
                                if contribution > 0
                                else "Negative"
                                if contribution < 0
                                else "Neutral"
                            )
                        })

                    shap_explanation.sort(
                        key=lambda x: abs(
                            x["contribution"]
                        ),
                        reverse=True
                    )

                except Exception as shap_error:

                    shap_explanation = []

                    print(
                        "SHAP Error:",
                        shap_error
                    )


                # -----------------------------------------
                # CALCULATE PRICE CHANGE
                # -----------------------------------------

                price_change = (
                    (
                        prediction
                        - current_price
                    )
                    / current_price
                ) * 100


                # -----------------------------------------
                # DECISION SUPPORT LOGIC
                # -----------------------------------------

                if price_change >= 5:

                    recommendation = (
                        "BUY EARLY"
                    )

                    explanation = (
                        "The predicted iron ore price "
                        "is expected to increase by 5% "
                        "or more. Early procurement may "
                        "help reduce exposure to a "
                        "potential future price increase."
                    )


                elif price_change <= -5:

                    recommendation = (
                        "WAIT"
                    )

                    explanation = (
                        "The predicted iron ore price "
                        "is expected to decrease by 5% "
                        "or more. Procurement may be "
                        "delayed if operational "
                        "requirements allow."
                    )


                else:

                    recommendation = (
                        "NORMAL PROCUREMENT"
                    )

                    explanation = (
                        "The predicted price change is "
                        "within the normal range of -5% "
                        "to +5%. Regular procurement "
                        "planning can be considered."
                    )


                # -----------------------------------------
                # SAVE PREDICTION HISTORY
                # -----------------------------------------

                PredictionHistory.objects.create(

                    steel_production=
                        steel_production,

                    demand=
                        demand,

                    supply=
                        supply,

                    freight=
                        freight,

                    oil=
                        oil,

                    usd_cny=
                        usd_cny,

                    current_price=
                        current_price,

                    predicted_price=
                        prediction,

                    price_change=
                        price_change,

                    recommendation=
                        recommendation,

                    model_name=
                        model_name,
                )


            except Exception as e:

                form.add_error(
                    None,
                    f"Prediction error: {str(e)}"
                )


    # -----------------------------------------
    # CONTEXT
    # -----------------------------------------

    context = {

        "form":
            form,

        "prediction":
            prediction,

        "current_price":
            current_price,

        "price_change":
            price_change,

        "recommendation":
            recommendation,

        "explanation":
            explanation,

        "model_name":
            model_name,

        "shap_explanation":
            shap_explanation,
    }


    return render(
        request,
        "prediction/future_prediction.html",
        context
    )


# ==================================================
# ML DASHBOARD
# ==================================================

@login_required
def dashboard(request):

    # -----------------------------------------
    # MODEL PATH
    # -----------------------------------------

    model_path = (
        Path(settings.BASE_DIR)
        / "ml_models"
        / "iron_ore_model.pkl"
    )


    # -----------------------------------------
    # CHECK MODEL
    # -----------------------------------------

    if not model_path.exists():

        return render(
            request,
            "prediction/dashboard.html",
            {
                "error":
                    "Model not found. "
                    "Please run train_model.py first."
            }
        )


    # -----------------------------------------
    # LOAD MODEL
    # -----------------------------------------

    package = joblib.load(
        model_path
    )


    # -----------------------------------------
    # RESULTS
    # -----------------------------------------

    results = package[
        "results"
    ]

    best_model_name = package[
        "model_name"
    ]


    # -----------------------------------------
    # MODEL TABLE
    # -----------------------------------------

    model_table = []

    for name, metrics in results.items():

        model_table.append({

            # IMPORTANT:
            # Dashboard template uses row.model
            "model":
                name,

            "mae":
                metrics["MAE"],

            "mse":
                metrics["MSE"],

            "rmse":
                metrics["RMSE"],

            "r2":
                metrics["R2"],
        })


    # -----------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------

    prediction_data = package[
        "predictions"
    ]


    actual_values = [

        item["actual"]

        for item in prediction_data

    ]


    predicted_values = [

        item["predicted"]

        for item in prediction_data

    ]


    # -----------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------

    feature_importance = package.get(
        "feature_importance",
        {}
    )


    feature_names = list(
        feature_importance.keys()
    )


    feature_values = list(
        feature_importance.values()
    )


    # -----------------------------------------
    # SEND DATA TO DASHBOARD
    # -----------------------------------------

    context = {

        "model_table":
            model_table,

        "best_model":
            best_model_name,

        "actual_values":
            actual_values,

        "predicted_values":
            predicted_values,

        "feature_names":
            feature_names,

        "feature_values":
            feature_values,

    }


    return render(
        request,
        "prediction/dashboard.html",
        context
    )


# ==================================================
# PREDICTION HISTORY
# ==================================================

@login_required
def prediction_history(request):

    # -----------------------------------------
    # LOAD ALL PREDICTIONS
    # -----------------------------------------

    history = (
        PredictionHistory.objects
        .all()
        .order_by("-created_at")
    )


    # -----------------------------------------
    # TOTAL PREDICTIONS
    # -----------------------------------------

    total_predictions = (
        history.count()
    )


    # -----------------------------------------
    # AVERAGE PREDICTED PRICE
    # -----------------------------------------

    average_prediction = 0

    if total_predictions > 0:

        average_prediction = (
            sum(
                item.predicted_price
                for item in history
            )
            / total_predictions
        )


    # -----------------------------------------
    # BUY EARLY COUNT
    # -----------------------------------------

    buy_count = history.filter(
        recommendation="BUY EARLY"
    ).count()


    # -----------------------------------------
    # WAIT COUNT
    # -----------------------------------------

    wait_count = history.filter(
        recommendation="WAIT"
    ).count()


    # -----------------------------------------
    # CONTEXT
    # -----------------------------------------

    context = {

        "history":
            history,

        "average_prediction":
            average_prediction,

        "buy_count":
            buy_count,

        "wait_count":
            wait_count,

    }


    return render(
        request,
        "prediction/history.html",
        context
    )


# ==================================================
# EXPORT PREDICTION HISTORY AS CSV
# ==================================================

@login_required
def export_history_csv(request):

    history = (
        PredictionHistory.objects
        .all()
        .order_by("-created_at")
    )


    response = HttpResponse(
        content_type="text/csv"
    )


    response["Content-Disposition"] = (
        'attachment; '
        'filename="iron_ore_prediction_history.csv"'
    )


    writer = csv.writer(
        response
    )


    # -----------------------------------------
    # CSV HEADER
    # -----------------------------------------

    writer.writerow([

        "Date & Time",

        "Model",

        "Steel Production Index",

        "Demand Index",

        "Supply Index",

        "Freight Cost (USD/tonne)",

        "Oil Price (USD/barrel)",

        "USD/CNY Exchange Rate",

        "Current Price (USD/tonne)",

        "Predicted Price (USD/tonne)",

        "Price Change (%)",

        "Recommendation",

    ])


    # -----------------------------------------
    # CSV DATA
    # -----------------------------------------

    for item in history:

        writer.writerow([

            item.created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            item.model_name,

            item.steel_production,

            item.demand,

            item.supply,

            item.freight,

            item.oil,

            item.usd_cny,

            item.current_price,

            item.predicted_price,

            round(
                item.price_change,
                2
            ),

            item.recommendation,

        ])


    return response


# ==================================================
# EXPORT PREDICTION HISTORY AS PDF
# ==================================================

@login_required
def export_history_pdf(request):

    history = (
        PredictionHistory.objects
        .all()
        .order_by("-created_at")
    )


    response = HttpResponse(
        content_type="application/pdf"
    )


    response["Content-Disposition"] = (
        'attachment; '
        'filename="iron_ore_prediction_history.pdf"'
    )


    document = SimpleDocTemplate(

        response,

        pagesize=landscape(A4),

        rightMargin=20,

        leftMargin=20,

        topMargin=20,

        bottomMargin=20,

    )


    styles = getSampleStyleSheet()

    elements = []


    # -----------------------------------------
    # PDF TITLE
    # -----------------------------------------

    title = Paragraph(
        "Iron Ore AI - Prediction History",
        styles["Title"]
    )


    elements.append(
        title
    )


    elements.append(
        Spacer(
            1,
            15
        )
    )


    # -----------------------------------------
    # PDF TABLE
    # -----------------------------------------

    data = [

        [

            "Date & Time",

            "Model",

            "Current Price",

            "Predicted Price",

            "Change",

            "Recommendation",

        ]

    ]


    for item in history:

        data.append([

            item.created_at.strftime(
                "%d %b %Y %H:%M"
            ),

            item.model_name,

            f"${item.current_price:.2f}",

            f"${item.predicted_price:.2f}",

            f"{item.price_change:.2f}%",

            item.recommendation,

        ])


    table = Table(
        data,
        repeatRows=1
    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e3a8a")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, 0),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, 0),
                8
            ),

        ])

    )


    elements.append(
        table
    )


    document.build(
        elements
    )


    return response


# ==================================================
# UPLOAD DATASET + RETRAIN MODEL
# ==================================================

@login_required
@user_passes_test(is_admin)
def upload_dataset(request):

    error = None

    success = None


    # -----------------------------------------
    # FIRST PAGE LOAD
    # -----------------------------------------

    if request.method != "POST":

        form = CSVUploadForm()

        return render(
            request,
            "prediction/upload.html",
            {
                "form":
                    form,

                "error":
                    error,

                "success":
                    success,
            }
        )


    # -----------------------------------------
    # FORM
    # -----------------------------------------

    form = CSVUploadForm(
        request.POST,
        request.FILES
    )


    if not form.is_valid():

        return render(
            request,
            "prediction/upload.html",
            {
                "form":
                    form,

                "error":
                    "Please select a valid CSV file.",

                "success":
                    success,
            }
        )


    uploaded_file = request.FILES[
        "csv_file"
    ]


    # -----------------------------------------
    # CHECK FILE TYPE
    # -----------------------------------------

    if not uploaded_file.name.lower().endswith(
        ".csv"
    ):

        error = (
            "Please upload a CSV file."
        )

        return render(
            request,
            "prediction/upload.html",
            {
                "form":
                    form,

                "error":
                    error,

                "success":
                    success,
            }
        )


    try:

        # -----------------------------------------
        # READ CSV
        # -----------------------------------------

        df = pd.read_csv(
            uploaded_file
        )


        # -----------------------------------------
        # REQUIRED COLUMNS
        # -----------------------------------------

        required_columns = [

            "Date",

            "Iron_Ore_Price_USD_per_tonne",

            "Steel_Production_Index",

            "Demand_Index",

            "Supply_Index",

            "Freight_Cost_USD_per_tonne",

            "Oil_Price_USD_per_barrel",

            "USD_CNY_Exchange_Rate",

        ]


        # -----------------------------------------
        # CHECK COLUMNS
        # -----------------------------------------

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]


        if missing_columns:

            error = (
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

            return render(
                request,
                "prediction/upload.html",
                {
                    "form":
                        form,

                    "error":
                        error,

                    "success":
                        success,
                }
            )


        # -----------------------------------------
        # CHECK RECORD COUNT
        # -----------------------------------------

        if len(df) < 30:

            error = (
                "The dataset must contain "
                "at least 30 records."
            )

            return render(
                request,
                "prediction/upload.html",
                {
                    "form":
                        form,

                    "error":
                        error,

                    "success":
                        success,
                }
            )


        # -----------------------------------------
        # VALIDATE DATE
        # -----------------------------------------

        df["Date"] = pd.to_datetime(

            df["Date"],

            errors="coerce",

            dayfirst=True
        )


        invalid_dates = (
            df["Date"].isna().sum()
        )


        if invalid_dates > 0:

            error = (

                "The Date column contains "

                f"{invalid_dates} invalid "
                "date value(s). "

                "Please check the Date column."

            )

            return render(
                request,
                "prediction/upload.html",
                {
                    "form":
                        form,

                    "error":
                        error,

                    "success":
                        success,
                }
            )


        # -----------------------------------------
        # SORT BY DATE
        # -----------------------------------------

        df = df.sort_values(
            "Date"
        )


        # -----------------------------------------
        # NUMERIC COLUMNS
        # -----------------------------------------

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


        # -----------------------------------------
        # CHECK NUMERIC VALUES
        # -----------------------------------------

        invalid_numeric_values = (

            df[numeric_columns]
            .isna()
            .sum()
            .sum()
        )


        if invalid_numeric_values > 0:

            error = (

                "Some numeric columns contain "

                f"{invalid_numeric_values} "

                "invalid or empty value(s)."

            )

            return render(
                request,
                "prediction/upload.html",
                {
                    "form":
                        form,

                    "error":
                        error,

                    "success":
                        success,
                }
            )


        # -----------------------------------------
        # DATASET PATH
        # -----------------------------------------

        dataset_path = (

            Path(settings.BASE_DIR)

            / "dataset"

            / "iron_ore_price_dataset_without_lag.csv"

        )


        # -----------------------------------------
        # BACKUP PATH
        # -----------------------------------------

        backup_path = (

            Path(settings.BASE_DIR)

            / "dataset"

            / "iron_ore_price_dataset_backup.csv"

        )


        # -----------------------------------------
        # CREATE BACKUP
        # -----------------------------------------

        if dataset_path.exists():

            shutil.copy2(

                dataset_path,

                backup_path

            )


        # -----------------------------------------
        # SAVE NEW DATASET
        # -----------------------------------------

        df.to_csv(

            dataset_path,

            index=False

        )


        # -----------------------------------------
        # RETRAIN MODEL
        # -----------------------------------------

        try:

            import train_model

            result = (
                train_model.train_model()
            )


            # -----------------------------------------
            # SUCCESS
            # -----------------------------------------

            success = (

                "Dataset uploaded and "
                "model retrained successfully. "

                f"Best Model: "
                f"{result['best_model']}. "

                f"Records Used: "
                f"{result['records']}."

            )


        except Exception as training_error:

            # -----------------------------------------
            # RESTORE OLD DATASET
            # -----------------------------------------

            if backup_path.exists():

                shutil.copy2(

                    backup_path,

                    dataset_path

                )


            error = (

                "Model retraining failed. "

                "The previous dataset has been "
                "restored. "

                f"Error: {training_error}"

            )


    except Exception as e:

        error = (

            "Unable to process the CSV file: "

            f"{e}"

        )


    # -----------------------------------------
    # RETURN PAGE
    # -----------------------------------------

    return render(
        request,
        "prediction/upload.html",
        {
            "form":
                form,

            "error":
                error,

            "success":
                success,
        }
    )