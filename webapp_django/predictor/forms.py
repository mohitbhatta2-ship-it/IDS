from django import forms

from . import ml


def _model_choices():
    return [(m["key"], m["name"]) for m in ml.available_models()]


class ManualFlowForm(forms.Form):
    """
    The 30 training features as numeric inputs.

    Fields are declared dynamically from ``ml.FEATURES`` rather than written out
    by hand, so the form can never drift out of sync with the feature list the
    models were trained on.
    """

    model_key = forms.ChoiceField(choices=_model_choices, required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_key"].choices = _model_choices()

        for feature in ml.FEATURES:
            stats = ml.FEATURE_STATS[feature]
            self.fields[feature] = forms.FloatField(
                required=True,
                label=feature,
                widget=forms.NumberInput(
                    attrs={
                        "step": "1" if stats["integer"] else "any",
                        "placeholder": _format_number(stats["median"]),
                    }
                ),
            )

    def clean_model_key(self):
        key = self.cleaned_data.get("model_key") or ml.DEFAULT_MODEL
        if key not in ml.MODEL_REGISTRY:
            raise forms.ValidationError("Unknown model.")
        return key

    def feature_values(self) -> dict:
        return {f: self.cleaned_data[f] for f in ml.FEATURES}


class BatchUploadForm(forms.Form):
    """Upload a CSV or Parquet file holding the 30 feature columns."""

    dataset = forms.FileField(
        label="Dataset file",
        help_text="CSV or Parquet containing the 30 feature columns. "
        "Include a Label column to also get accuracy and F1.",
    )
    model_key = forms.ChoiceField(choices=_model_choices, required=False, label="Model")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_key"].choices = _model_choices()

    def clean_model_key(self):
        key = self.cleaned_data.get("model_key") or ml.DEFAULT_MODEL
        if key not in ml.MODEL_REGISTRY:
            raise forms.ValidationError("Unknown model.")
        return key

    def clean_dataset(self):
        f = self.cleaned_data["dataset"]
        name = f.name.lower()
        if not name.endswith((".csv", ".parquet")):
            raise forms.ValidationError("Please upload a .csv or .parquet file.")
        return f


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"
