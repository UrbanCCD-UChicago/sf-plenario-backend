from wopr import create_app
from werkzeug.contrib.fixers import ProxyFix
#celery_app = make_celery(app=app)

app = create_app()
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    #app.run(debug=True, use_reloader=False)
    app.run()
