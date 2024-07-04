from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze_neck/', views.analyze_neck, name='analyze_neck'),
    path('analyze_back/', views.analyze_back, name='analyze_back'),
]