from django import forms


class FuturePredictionForm(forms.Form):

    steel_production = forms.FloatField(
        label="Steel Production Index",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter steel production index",
        }),
    )

    demand = forms.FloatField(
        label="Demand Index",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter demand index",
        }),
    )

    supply = forms.FloatField(
        label="Supply Index",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter supply index",
        }),
    )

    freight = forms.FloatField(
        label="Freight Cost (USD/tonne)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter freight cost",
        }),
    )

    oil = forms.FloatField(
        label="Oil Price (USD/barrel)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter oil price",
        }),
    )

    usd_cny = forms.FloatField(
        label="USD/CNY Exchange Rate",
        min_value=0,
        widget=forms.NumberInput(attrs={
            "class": "form-input",
            "step": "any",
            "placeholder": "Enter USD/CNY exchange rate",
        }),
    )


class CSVUploadForm(forms.Form):

    csv_file = forms.FileField(
        label="Upload Iron Ore Dataset",
        widget=forms.ClearableFileInput(attrs={
            "class": "form-input",
            "accept": ".csv",
        }),
    )