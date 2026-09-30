from django.urls import path
from . import views

urlpatterns = [

    path(
        "",
        views.home,
        name="home"
    ),

    path(
        "future-prediction/",
        views.future_prediction,
        name="future_prediction"
    ),

    path(
        "dashboard/",
        views.dashboard,
        name="dashboard"
    ),

    path(
        "history/",
        views.prediction_history,
        name="prediction_history"
    ),

    path(
        "history/export/csv/",
        views.export_history_csv,
        name="export_history_csv"
    ),

    path(
        "history/export/pdf/",
        views.export_history_pdf,
        name="export_history_pdf"
    ),

    path(
        "upload/",
        views.upload_dataset,
        name="upload_dataset"
    ),
]