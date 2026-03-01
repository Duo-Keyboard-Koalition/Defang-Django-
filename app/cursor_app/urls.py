from django.urls import path
from . import views

app_name = 'cursor_app'

urlpatterns = [
    path('v1/models', views.models, name='models'),
    path('v1/chat/completions', views.chat_completions, name='chat_completions'),
    path('health/', views.health, name='health'),
]
