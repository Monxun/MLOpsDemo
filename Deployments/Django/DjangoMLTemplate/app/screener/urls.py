from django.urls import path
from . import views
from .api import api

# {% url 'app_name:name' %}
app_name = 'screener'

urlpatterns = [
    path('', views.screener, name='screener'), # Screener page
    path('api_v1/', api.urls), # Connects to api.py
]