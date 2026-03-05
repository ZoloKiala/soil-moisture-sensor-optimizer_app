# sensor_optimizer/urls.py
from django.urls import path
from . import views

app_name = "sensor_optimizer"

urlpatterns = [
    # Form (GET) + Run optimization (POST)
    path("sensor-optimizer/", views.sensor_optimizer_view, name="sensor_optimizer"),

    # ✅ Results should be GET (prevents "Resubmit the form?")
    path(
        "sensor-optimizer/results/<str:run_id>/",
        views.results_view,
        name="sensor_optimizer_results",
    ),

    # Keep your existing map endpoint (POST-based or session-based)
    path("sensor-optimizer/map/", views.centroid_map_view, name="sensor_optimizer_map"),

    # Optional (recommended): run-based map as GET too
    # path(
    #     "sensor-optimizer/results/<str:run_id>/map/",
    #     views.centroid_map_view,
    #     name="sensor_optimizer_map_run",
    # ),

    path("sensor-optimizer/report/", views.download_layout_report_view, name="download_layout_report"),
path("sensor-optimizer/feedback/ajax/", views.feedback_ajax_view, name="feedback_ajax"),
]