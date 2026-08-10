from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.manual, name="manual"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("batch/", views.batch, name="batch"),
    path("live/", views.live, name="live"),
    path("live/start/", views.api_live_start, name="api_live_start"),
    path("live/stop/", views.api_live_stop, name="api_live_stop"),
    path("live/status/", views.api_live_status, name="api_live_status"),
    path("history/", views.history, name="history"),
    path("history/download/", views.download_history, name="download_history"),
    path("history/clear/", views.clear_history, name="clear_history"),
]
