from django.apps import AppConfig


class PredictorConfig(AppConfig):
    name = 'predictor'

    def ready(self):
        # Install the unified live-capture session so the existing live console transparently
        # routes cleartext FTP flows to the validated FTP detector while non-FTP/FTPS flows keep
        # using the selected production model. Import side-effect installs the factory; wrapped
        # so app startup never fails if the FTP integration module cannot import.
        try:
            from . import ftp_live  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
