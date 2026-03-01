from django.urls import path
from . import views

app_name = 'cortex_app'

urlpatterns = [
    path('chat/', views.cortex_chat, name='chat'),
    path('health/', views.cortex_health, name='health'),
]
