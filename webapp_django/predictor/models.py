from django.db import models


class PredictionLog(models.Model):
    """
    One row per prediction run, so the app keeps a history rather than being
    stateless. Manual runs record the flow that was classified; batch runs
    record file-level totals and, when the upload carried a Label column, the
    evaluation scores as well.
    """

    MANUAL = "manual"
    BATCH = "batch"
    KIND_CHOICES = [(MANUAL, "Manual entry"), (BATCH, "Dataset upload")]

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    kind = models.CharField(max_length=10, choices=KIND_CHOICES, db_index=True)

    model_key = models.CharField(max_length=40)
    model_name = models.CharField(max_length=60)

    # Manual entry
    predicted_label = models.CharField(max_length=60, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    features = models.JSONField(null=True, blank=True)

    # Dataset upload
    source_filename = models.CharField(max_length=255, blank=True)
    row_count = models.PositiveIntegerField(null=True, blank=True)
    attack_count = models.PositiveIntegerField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    macro_f1 = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "prediction"
        verbose_name_plural = "predictions"

    def __str__(self):
        if self.kind == self.MANUAL:
            return f"{self.predicted_label} ({self.model_name})"
        return f"{self.source_filename} — {self.row_count} rows ({self.model_name})"

    @property
    def is_attack(self) -> bool:
        return bool(self.predicted_label) and self.predicted_label != "Benign"

    @property
    def was_evaluated(self) -> bool:
        """
        True when the upload carried a usable Label column, so accuracy and
        macro F1 were computed. Checked explicitly against None because a
        genuinely poor run can score 0.0, which is a real result rather than
        a missing one.
        """
        return self.accuracy is not None

    @property
    def blank_reason(self) -> str:
        """Why this run has no evaluation scores, phrased for a tooltip."""
        if self.kind == self.MANUAL:
            return (
                "Manual entries classify a single flow with no ground truth, "
                "so there is nothing to score against."
            )
        return (
            "This upload had no Label column, so there was no ground truth to "
            "score the predictions against. Re-upload a file that includes one "
            "to get accuracy and macro F1."
        )

    @property
    def attack_share(self) -> float:
        if not self.row_count:
            return 0.0
        return (self.attack_count or 0) / self.row_count
