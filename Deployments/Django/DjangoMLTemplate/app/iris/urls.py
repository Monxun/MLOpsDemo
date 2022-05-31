from django.urls import path
from . import views
from .api import api

# {% url 'app_name:name' %}
app_name = 'iris'

urlpatterns = [
    path('', views.predict, name='predict'), # Prediction form page
    path('predict/', views.predict_chances, name='submit_prediction'),
    path('results/', views.results, name='results'), # Prediction form page
    path('api_v1/', api.urls), # Connects to api.py
]