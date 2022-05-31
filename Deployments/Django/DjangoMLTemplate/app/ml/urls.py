from django.urls import path
from . import views
from .api import api

# {% url 'app_name:name' %}
app_name = 'ml'

urlpatterns = [
    path('', views.overview, name='overview'), # ML home page
    path('api_v1/', api.urls), # Connects to api.py
]