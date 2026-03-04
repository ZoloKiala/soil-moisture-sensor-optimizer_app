# sensor_optimizer/urls.py
from django.urls import path
from . import views

app_name = "sensor_optimizer"

urlpatterns = [
    path("sensor-optimizer/", views.sensor_optimizer_view, name="sensor_optimizer"),
    path("sensor-optimizer/map/", views.centroid_map_view, name="sensor_optimizer_map"),
    path("sensor-optimizer/report/", views.download_layout_report_view, name="download_layout_report"),
]