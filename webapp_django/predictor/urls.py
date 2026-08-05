from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.manual, name="manual"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("batch/", views.batch, name="batch"),
    path("history/", views.history, name="history"),
    path("history/clear/", views.clear_history, name="clear_history"),
]
