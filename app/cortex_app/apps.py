from django.apps import AppConfig


class CortexAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cortex_app'
    verbose_name = 'Cortex (OpenAI Wrapper)'
