from wopr import create_app
from werkzeug.contrib.fixers import ProxyFix
#celery_app = make_celery(app=app)

app = create_app()
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('.logs/app.log', maxBytes=10**6)
    app.logger.addHandler(file_handler)
    app.run(host='localhost', port=8000, debug=True)
