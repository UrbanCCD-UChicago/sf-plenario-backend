from wopr import create_app
from werkzeug.contrib.fixers import ProxyFix
#celery_app = make_celery(app=app)

app = create_app()
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == "__main__":
    if not app.debug:
        import logging
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler('.logs/app.log', maxBytes=10**6)
        file_handler.setLevel(logging.DEBUG)
        app.logger.addHandler(file_handler)
    #app.run(host='0.0.0.0', port=80)
    app.run(host='localhost', port=8000)
