"""
URL configuration for soilmoisture_site project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.shortcuts import redirect
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static



def root_redirect(request):
    return redirect("/sensor-optimizer/")

urlpatterns = [
    path("", root_redirect),  # homepage redirect
    path("", include(("sensor_optimizer.urls", "sensor_optimizer"), namespace="sensor_optimizer")),
    path("favicon.ico", RedirectView.as_view(
        url="/static/sensor_optimizer/favicon.ico",
        permanent=True
    )),
]


handler404 = "soilmoisture_site.views.error_404"
handler500 = "soilmoisture_site.views.error_500"



if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


