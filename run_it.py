from wopr import create_app
#celery_app = make_celery(app=app)

proj_str = '+proj=lcc +lat_1=37.06666666666667 +lat_2=38.43333333333333 \
+lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000.0000000002 +datum=NAD83 \
+units=us-ft +no_defs'

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
