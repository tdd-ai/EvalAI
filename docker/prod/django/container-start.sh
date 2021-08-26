#!/bin/sh
python manage.py migrate --noinput  && \
python manage.py collectstatic --noinput  && \
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser('$SUPERUSER', '$SUPERUSEREMAIL', '$SUPERPASSWORD')" | python manage.py shell && \
uwsgi --ini /code/docker/prod/django/uwsgi.ini
