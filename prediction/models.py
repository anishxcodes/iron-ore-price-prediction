from django.db import models


class PredictionHistory(models.Model):

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    steel_production = models.FloatField()

    demand = models.FloatField()

    supply = models.FloatField()

    freight = models.FloatField()

    oil = models.FloatField()

    usd_cny = models.FloatField()

    current_price = models.FloatField()

    predicted_price = models.FloatField()

    price_change = models.FloatField()

    recommendation = models.CharField(
        max_length=100
    )

    model_name = models.CharField(
        max_length=100
    )


    def __str__(self):

        return (
            f"{self.created_at} - "
            f"{self.predicted_price}"
        )