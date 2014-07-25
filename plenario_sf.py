from wopr import create_app
#celery_app = make_celery(app=app)

app = create_app()

if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    app.run(debug=True, use_reloader=False)
