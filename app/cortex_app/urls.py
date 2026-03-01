from django.urls import path
from . import views

app_name = 'cortex_app'

urlpatterns = [
    # OpenAI-compatible (use this base URL in Cursor: /cortex/v1)
    path('v1/chat/completions', views.chat_completions, name='chat_completions'),
    path('v1/models',           views.models_list,      name='models_list'),

    # Simple shorthand
    path('chat/',               views.cortex_chat,      name='chat'),
    path('health/',             views.cortex_health,    name='health'),
]
