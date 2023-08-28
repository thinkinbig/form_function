gunicorn --pythonpath $(pwd) --workers 8 --bind 0.0.0.0:3490 --log-level=debug \
--name gunicorn wsgi:app
# --daemon --pid /tmp/gunicorn.pid \
# --name gunicorn --chdir $(pwd) wsgi:app

