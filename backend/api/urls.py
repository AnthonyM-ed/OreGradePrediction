# api/urls.py (URLs de la aplicación API)
from django.urls import path
from . import views

urlpatterns = [
    path('config/', views.get_config, name='get_config'),
    path('table/<str:table_name>/', views.get_table_data, name='get_table_data'),
    path('heatmap/', views.get_heatmap, name='get_heatmap'),
    path('ask/', views.ask_question, name='ask_question'),
    path('predict-ore-grade/', views.predict_ore_grade, name='predict_ore_grade'),
    path('available-models/', views.get_available_models, name='get_available_models'),
]