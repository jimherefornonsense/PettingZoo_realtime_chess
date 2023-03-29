from django.urls import path
from . import views


urlpatterns = [
    path('upload/', views.create_my_model, name='create-my-model'),
    path('board/', views.get_my_models_grouped_by_uuid,
    name='get_my_models_grouped_by_uuid'),
]
