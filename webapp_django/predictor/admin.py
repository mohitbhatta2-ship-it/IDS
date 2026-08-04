from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at", "kind", "model_name",
        "predicted_label", "source_filename", "row_count", "macro_f1",
    )
    list_filter = ("kind", "model_name", "predicted_label")
    search_fields = ("predicted_label", "source_filename")
    readonly_fields = tuple(f.name for f in PredictionLog._meta.fields)
