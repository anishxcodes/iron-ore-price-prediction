from django.contrib import admin

from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):

    list_display = (
        "created_at",
        "model_name",
        "current_price",
        "predicted_price",
        "price_change",
        "recommendation",
    )

    list_filter = (
        "model_name",
        "recommendation",
    )

    search_fields = (
        "model_name",
        "recommendation",
    )

    ordering = (
        "-created_at",
    )