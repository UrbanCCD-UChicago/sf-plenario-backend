from wopr import create_app
#celery_app = make_celery(app=app)

plenario_sf = create_app()

if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    plenario_sf.run(debug=True, use_reloader=False)
