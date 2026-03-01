from django.urls import path
from . import views

app_name = 'frontend_app'

urlpatterns = [
    path('', views.chat_ui, name='chat_ui'),
    path('api/chat/', views.chat_proxy, name='chat_proxy'),
    path('api/models/', views.models_list, name='models_list'),
]
