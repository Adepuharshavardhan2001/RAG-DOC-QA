import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from api.evaluation import evaluate

# Replace 1 with your actual user ID
evaluate(user_id=3)