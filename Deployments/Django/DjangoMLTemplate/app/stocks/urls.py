from django.urls import path
from . import views
from .api import api

app_name = 'stocks'

urlpatterns = [
    path('', views.index, name='index'), # Stocks page
    path('<str:tid>', views.ticker, name='ticker'), # Ticker id info view
    path('api_v1/', api.urls) # Connects to api.py
]