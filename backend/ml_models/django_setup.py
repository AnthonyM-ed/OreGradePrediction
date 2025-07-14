"""
Django Setup Utility
====================

Ensures Django is properly configured before any ml_models modules are imported.
This must be called before importing any modules that access Django settings.
"""

import os
import django
from django.conf import settings


def setup_django():
    """
    Setup Django configuration if not already configured.
    This should be called before importing any modules that access Django settings.
    """
    if not settings.configured:
        # Set the Django settings module
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
        
        # Configure Django
        django.setup()


def ensure_django_configured():
    """
    Ensure Django is configured, raising an error if not.
    """
    if not settings.configured:
        raise RuntimeError(
            "Django settings not configured. Call setup_django() before importing ml_models modules."
        )


# Auto-setup Django when this module is imported
setup_django()
