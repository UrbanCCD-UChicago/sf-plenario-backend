from flask import make_response, request, render_template, current_app, g, \
    Blueprint
from functools import update_wrapper
import os
import math
from datetime import date, datetime, timedelta
from datetime_truncate import truncate
import calendar
import time
import json
from sqlalchemy import func, case, distinct, Column, Float, Table
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.types import NullType
from sqlalchemy.sql.expression import cast,\
        Executable, ClauseElement,_literal_as_text
from sqlalchemy.ext.compiler import compiles
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON
from operator import itemgetter
from itertools import groupby
from cStringIO import StringIO
import csv
from shapely.wkb import loads
from shapely.geometry import box, asShape
from collections import OrderedDict

from wopr.models import MasterTable, MetaTable
from wopr.database import session, app_engine as engine, Base

import time

api = Blueprint('api', __name__)

dthandler = lambda obj: obj.isoformat() if isinstance(obj, date) else None

query_types = {
    'area':     {'func': lambda **kwargs: area(**kwargs)     },
    'count':    {'func': lambda **kwargs: count(**kwargs)    },
    'weighted': {'func': lambda **kwargs: weighted(**kwargs) },
    'dist':     {'func': lambda **kwargs: dist(**kwargs)     }
}

def increment_datetime(sourcedate, time_agg):
    if time_agg == 'day':
        days_to_add = 1
    if time_agg == 'week':
        days_to_add = 7
    elif time_agg == 'month':
        _, days_to_add = calendar.monthrange(sourcedate.year, sourcedate.month)
    elif time_agg == 'year':
        days_to_add = 366 if calendar.isleap(sourcedate.year) else 365
    return sourcedate + timedelta(days=days_to_add)


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True): # pragma: no cover
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

def make_query(table, raw_query_params, resp=None):
    table_keys = table.columns.keys()
    args_keys = raw_query_params.keys()
    if not resp:
        resp = {
            'meta': {
                'status': 'error',
                'message': '',
            },
            'objects': [],
        }
    status_code = 200
    query_clauses = []
    valid_query = True
    if 'offset' in args_keys:
        args_keys.remove('offset')
    if 'limit' in args_keys:
        args_keys.remove('limit')
    if 'order_by' in args_keys:
        args_keys.remove('order_by')
    for query_param in args_keys:
        try:
            field, operator = query_param.split('__')
        except ValueError:
            field = query_param
            operator = 'eq'
        query_value = raw_query_params.get(query_param)
        column = table.columns.get(field)
        if field not in table_keys:
            resp['meta']['message'] = '"%s" is not a valid fieldname' % field
            status_code = 400
            valid_query = False
        elif operator == 'in':
            query = column.in_(query_value.split(','))
            query_clauses.append(query)
        elif operator in ['within', 'intersects']:
            geo = json.loads(query_value)
            if 'features' in geo.keys():
                val = geo['features'][0]['geometry']
            elif 'geometry' in geo.keys():
                val = geo['geometry']
            else:
                val = geo
            if val['type'] == 'LineString':
                shape = asShape(val)
                lat = shape.centroid.y
                # 100 meters by default
                x, y = getSizeInDegrees(100, lat)
                val = shape.buffer(y).__geo_interface__
            val['crs'] = {"type":"name","properties":{"name":"EPSG:4326"}}
            if operator == 'within':
                query = column.ST_Within(func.ST_GeomFromGeoJSON(json.dumps(val)))
            elif operator == 'intersects':
                query =\
                    column.ST_Intersects(func.ST_GeomFromGeoJSON(json.dumps(val)))
            query_clauses.append(query)
        elif operator.startswith('time_of_day'):
            if operator.endswith('ge'):
                query = func.date_part('hour', column).__ge__(query_value)
            elif operator.endswith('le'):
                query = func.date_part('hour', column).__le__(query_value)
            query_clauses.append(query)
        else:
            try:
                attr = filter(
                    lambda e: hasattr(column, e % operator),
                    ['%s', '%s_', '__%s__']
                )[0] % operator
            except IndexError:
                resp['meta']['message'] = '"%s" is not a valid query operator' % operator
                status_code = 400
                valid_query = False
                break
            if query_value == 'null': # pragma: no cover
                query_value = None
            query = getattr(column, attr)(query_value)
            query_clauses.append(query)
    return valid_query, query_clauses, resp, status_code

@api.route('/api/')
@crossdomain(origin="*")
def meta():
    status_code = 200
    resp = []
    dataset_name = request.args.get('dataset_name')
    if dataset_name:
        values = session.query(MetaTable)\
            .filter(MetaTable.c.dataset_name == dataset_name).all()
    else:
        values = session.query(MetaTable).all()
    keys = MetaTable.columns.keys()
    for value in values:
        d = {}
        for k,v in zip(keys, value):
            d[k] = v
        resp.append(d)
    resp = make_response(json.dumps(resp, default=dthandler), status_code)
    resp.headers['Content-Type'] = 'application/json'
    return resp

@api.route('/api/fields/<dataset_name>/')
@crossdomain(origin="*")
def dataset_fields(dataset_name):
    try:
        table = Table('dat_%s' % dataset_name, Base.metadata,
            autoload=True, autoload_with=engine,
            extend_existing=True)
        data = {
            'meta': {
                'status': 'ok',
                'message': ''
            },
            'objects': []
        }
        status_code = 200
        table_exists = True
    except NoSuchTableError:
        table_exists = False
        data = {
            'meta': {
                'status': 'error',
                'message': "'%s' is not a valid table name" % dataset_name
            },
            'objects': []
        }
        status_code = 400
    if table_exists:
        fields = table.columns.keys()
        for col in table.columns:
            if not isinstance(col.type, NullType):
                d = {}
                d['field_name'] = col.name
                d['field_type'] = str(col.type)
                data['objects'].append(d)
    resp = make_response(json.dumps(data), status_code)
    resp.headers['Content-Type'] = 'application/json'
    return resp

def make_csv(data):
    outp = StringIO()
    writer = csv.writer(outp)
    writer.writerows(data)
    return outp.getvalue()
            
def contour_intersect(query_geom=None, contour_table_name='sf_shore',
    buffer_radius=0):
    """
    Using a table containing a shape representing the general area under
    consideration (e.g. a city, a state, etc) return its intersection with the
    geometry query (if provided.)
    """
    # Retrieve the shore contours to consider land only
    land_table = Table(contour_table_name, Base.metadata,
        autoload=True, autoload_with=engine)
    # If a query geometry is provided, compute its intersection
    # with the shore contours; otherwise, just use the land
    # contours
    if query_geom:
        hot_geom = case([(
                func.ST_Within(func.ST_GeomFromGeoJSON(query_geom),
                        land_table.c['geom']),
                func.ST_GeomFromGeoJSON(query_geom)
            )],
            else_=\
                func.ST_Intersection(func.ST_GeomFromGeoJSON(query_geom),
                                     land_table.c['geom'])
        )
        #hot_geom = func.ST_Intersection(
        #    func.ST_GeomFromGeoJSON(query_geom),
        #    #func.ST_buffer(land_table.c['geom'], buffer_radius)
        #    land_table.c['geom']
        #)
    else:
        hot_geom = land_table.c['geom']
    land_val = session.query(func.ST_AsGeoJSON(hot_geom)).first()[0]
    land_val = json.loads(land_val)
    land_val['crs'] = {"type":"name","properties":{"name":"EPSG:4326"}}
    land_geom = json.dumps(land_val)
    return land_geom

@api.route('/api/indicators/types/<query_type>/')
@crossdomain(origin='*')
def indicators_by_type(query_type):
    resp, status_code = query_types[query_type]['func']()
    resp = make_response(json.dumps(resp, default=dthandler), status_code)
    resp.headers['Content-Type'] = 'application/json'
    return resp

def area(land_only=True):
    raw_query_params = request.args.copy()
    dataset_name = None
    if 'dataset_name' in raw_query_params.keys():
        dataset_name = raw_query_params['dataset_name']
        del raw_query_params['dataset_name']
    if 'obs_date__ge' in raw_query_params.keys():
        raw_query_params['end_date__ge'] = raw_query_params['obs_date__ge']
        del raw_query_params['obs_date__ge']
        from_date = datetime.strptime(raw_query_params['end_date__ge'], '%Y/%m/%d')
    if 'obs_date__le' in raw_query_params.keys():
        raw_query_params['start_date__le'] = raw_query_params['obs_date__le']
        del raw_query_params['obs_date__le']
        to_date = datetime.strptime(raw_query_params['start_date__le'], '%Y/%m/%d')
    if 'agg' in raw_query_params.keys():
        agg = raw_query_params['agg']
        del raw_query_params['agg']
    else:
        agg = 'day'
    if 'location_geom__within' in raw_query_params.keys():
        raw_query_params['geom__intersects'] = raw_query_params['location_geom__within']
        del raw_query_params['location_geom__within']
        val = json.loads(raw_query_params['geom__intersects'])['geometry']
        val['crs'] = {"type":"name", "properties":{"name":"EPSG:4326"}}
        query_geom = json.dumps(val)
    else:
        query_geom = None
    # Pull data from meta_table
    meta_table = Table('sf_meta', Base.metadata, autoload=True,
        autoload_with=engine)
    meta_query = session.query(
        meta_table.c['table_name'],
        meta_table.c['human_name']
    ).filter('area_q')
    if dataset_name:
        meta_query = meta_query.filter(meta_table.c['table_name'] == dataset_name)
    datasets = meta_query.all()
    resp = {
        'meta': {
            'status': 'ok',
            'message': '',
        },
        'objects': [],
    }
    status_code = 200
    for dataset in datasets:
        table_name = dataset[0]
        human_name = dataset[1]
        table = Table(table_name, Base.metadata,
            autoload=True, autoload_with=engine)
        valid_query, query_clauses, resp, status_code =\
            make_query(table, raw_query_params, resp)
        if valid_query:
            if land_only:
                # Retrieve the shore contours to consider land only
                land_table = Table('sf_shore', Base.metadata,
                    autoload=True, autoload_with=engine)
                # If a query geometry is provided, compute its intersection
                # with the shore contours; otherwise, just use the land
                # contours
                if query_geom:
                    hot_geom = func.ST_Intersection(
                        func.ST_GeomFromGeoJSON(query_geom),
                        land_table.c['geom']
                    )
                else:
                    hot_geom = land_table.c['geom']
                land_val = session.query(func.ST_AsGeoJSON(hot_geom)).first()[0]
                land_val = json.loads(land_val)
                land_val['crs'] = {"type":"name","properties":{"name":"EPSG:4326"}}
                land_geom = json.dumps(land_val)
            else:
                land_geom = query_geom
            if query_geom:
                # query_geom is used instead of land_geom to compute
                # intersections since the geometries in the queried dataset are
                # all on land, and query geom is usually a simpler geometry
                # than land_geom.
                # In order to avoid cumputing unneccesary intersection, we
                # check if a feature is entirely included in the query
                # geometry using the SQL case statement.
                hot_geom = case([(
                        func.ST_Within(table.c['geom'],
                                       func.ST_GeomFromGeoJSON(query_geom)),
                        table.c['geom']
                    )],
                    else_=\
                        func.ST_Intersection(func.ST_GeomFromGeoJSON(query_geom),
                                             table.c['geom'])
                )
            else:
                hot_geom = table.c['geom']
            #base_query = session.query(
            #    func.sum(func.ST_Area(hot_geom)) /\
            #        func.ST_Area(func.ST_GeomFromGeoJSON(land_geom))
            #)
            table_start_date = func.date_trunc(agg, table.c['start_date'])
            table_end_date = func.date_trunc(agg, table.c['end_date'])
            base_query = session.query(
                table_start_date, table_end_date,
                func.ST_Area(hot_geom) /\
                    func.ST_Area(func.ST_GeomFromGeoJSON(land_geom))
            )
            # Applying this filtering makes the query compute the actual
            # intersection only with polygons that actually intersects
            for clause in query_clauses:
                base_query = base_query.filter(clause)
            values = [v for v in base_query.all()]
            log = {}
            if not from_date:
                from_date = datetime(2000, 1, 1)
            if not to_date:
                to_date = datetime.now()
            # Create start and end log entries
            log[str(from_date.date())] = 0.0
            log[str(to_date.date())] = 0.0
            for v in values:
                start = v[0] if v[0] > from_date else from_date
                end = v[1]
                log[str(start.date())] =\
                    log.get(str(start.date()), 0.0) + v[2]
                if end <= to_date:
                    log[str(end.date())] =\
                        log.get(str(end.date()), 0.0) - v[2]
            log = OrderedDict(sorted(log.items()))
            cum_value = 0.0
            for k in log:
                cum_value = cum_value + log[k]
                log[k] = cum_value
            d = {
                'dataset_name': table_name, 
                'human_name': human_name,
                'query_type': 'area',
                'response_type': 'time-series',
                'time_agg': agg
            }
            return_values = []
            for k in log:
                return_values.append({
                    'date': datetime.strptime(k, '%Y-%m-%d'),
                    'value': round(log[k], 4)
                })
            d['values'] = return_values
            resp['objects'].append(d)
        else:
            resp['meta']['status'] = 'error'
            resp['meta']['message'] = 'Invalid query.'
            resp['objects'] = []
            break
    return resp, status_code
    
def count():
    raw_query_params = request.args.copy()
    dataset_name = None
    if 'dataset_name' in raw_query_params.keys():
        dataset_name = raw_query_params['dataset_name']
        del raw_query_params['dataset_name']
    if 'location_geom__within' in raw_query_params.keys():
        raw_query_params['geom__within'] = raw_query_params['location_geom__within']
        del raw_query_params['location_geom__within']
    if 'agg' in raw_query_params.keys():
        agg = raw_query_params['agg']
        del raw_query_params['agg']
    else:
        agg = 'day'
    if 'obs_date__ge' in raw_query_params.keys():
        from_date = datetime.strptime(raw_query_params['obs_date__ge'], '%Y/%m/%d')
    else:
        from_date = datetime(2000, 01, 01)
    from_date = truncate(from_date, agg)
    if 'obs_date__le' in raw_query_params.keys():
        to_date = datetime.strptime(raw_query_params['obs_date__le'], '%Y/%m/%d')
    else:
        to_date = datetime.now()
    to_date = truncate(to_date, agg)
    # Pull data from meta_table
    meta_table = Table('sf_meta', Base.metadata, autoload=True,
        autoload_with=engine)
    meta_query = session.query(
        meta_table.c['table_name'],
        meta_table.c['human_name'],
        meta_table.c['duration']
    ).filter('count_q')
    if dataset_name:
        meta_query = meta_query.filter(meta_table.c['table_name'] == dataset_name)
    datasets = meta_query.all()
    resp = {
        'meta': {
            'status': 'ok',
            'message': '',
        },
        'objects': [],
    }
    status_code = 200
    for dataset in datasets:
        table_name = dataset[0]
        human_name = dataset[1]
        duration = dataset[2]
        table = Table(table_name, Base.metadata,
            autoload=True, autoload_with=engine)
        local_params = raw_query_params.copy()
        if 'obs_date__ge' in local_params.keys() and duration == 'interval':
            local_params['end_date__ge'] = local_params['obs_date__ge']
            del local_params['obs_date__ge']
        if 'obs_date__le' in local_params.keys() and duration == 'interval':
            local_params['start_date__le'] = local_params['obs_date__le']
            del local_params['obs_date__le']
        valid_query, query_clauses, resp, status_code =\
            make_query(table, local_params, resp)
        if valid_query:
            if duration == 'interval':
                table_start_date = func.date_trunc(agg, table.c['start_date'])
                table_end_date = func.date_trunc(agg, table.c['end_date'])
                base_query = session.query(
                    table_start_date, table_end_date
                )
                for clause in query_clauses:
                    base_query = base_query.filter(clause)
                values = [v for v in base_query.all()]
                log = {}
                # Create start and end log entries
                log[from_date] = 0
                log[to_date] = 0
                for v in values:
                    start = v[0] if v[0] > from_date else from_date
                    end = v[1]
                    log[start] =\
                        log.get(start, 0) + 1
                    if end <= to_date:
                        log[end] =\
                            log.get(end, 0) - 1
                log = OrderedDict(sorted(log.items()))
                cum_value = 0
                for k in log:
                    cum_value = cum_value + log[k]
                    log[k] = cum_value
            else:
                table_date = func.date_trunc(agg, table.c['obs_date'])
                base_query = session.query(
                    table_date,
                    func.count(table.c['row_id'])
                ).group_by(table_date).order_by(table_date)
                for clause in query_clauses:
                    base_query = base_query.filter(clause)
                for clause in query_clauses:
                    base_query = base_query.filter(clause)
                values = [v for v in base_query.all()]
                # Need to fill in the missing values with zeros                
                filled_values = []
                cursor = from_date
                v_index = 0
                if len(values) > 0:
                    while v_index < len(values) and\
                            cursor > values[v_index][0].replace(tzinfo=None):
                        v_index += 1
                while cursor <= to_date:
                    if v_index < len(values) and\
                        values[v_index][0].replace(tzinfo=None) == cursor:
                        filled_values.append(values[v_index])
                        v_index += 1
                    else:
                        filled_values.append((cursor, 0))
                    cursor = increment_datetime(cursor, agg)
                log = OrderedDict(filled_values)
            d = {
                'dataset_name': table_name,
                'human_name': human_name,
                'query_type': 'count',
                'response_type': 'time-series',
                'time_agg': agg
            }
            return_values = []
            for k in log:
                return_values.append({
                    'date': k,
                    'value': log[k]
                })
            d['values'] = return_values
            resp['objects'].append(d)
        else:
            resp['meta']['status'] = 'error'
            resp['meta']['message'] = 'Invalid query.'
            resp['objects'] = []
            break
    return resp, status_code

class explain(Executable, ClauseElement):
    def __init__(self, stmt, analyze=False):
        self.statement = _literal_as_text(stmt)
        self.analyze = analyze

@compiles(explain, 'postgresql')
def pg_explain(element, compiler, **kw):
    text = "EXPLAIN "
    if element.analyze:
        text += "ANALYZE "
    text += compiler.process(element.statement)
    return text

def dist():
    """
    This type of queries "discretizes" the geometry into census blocks, and
    computes the average (or weighted) distance to the closest point of
    interest in the dataset.
    """
    raw_query_params = request.args.copy()
    blocks_raw_query_params = {}
    dataset_name = None
    query_geom = None
    if 'dataset_name' in raw_query_params.keys():
        dataset_name = raw_query_params['dataset_name']
        del raw_query_params['dataset_name']
    if 'location_geom__within' in raw_query_params.keys():
        blocks_raw_query_params['centroid__within'] = raw_query_params['location_geom__within']
        del raw_query_params['location_geom__within']
        val = json.loads(blocks_raw_query_params['centroid__within'])['geometry']
        val['crs'] = {"type":"name", "properties":{"name":"EPSG:4326"}}
        query_geom = json.dumps(val)
    if 'agg' in raw_query_params.keys():
        agg = raw_query_params['agg']
        del raw_query_params['agg']
    else:
        agg = 'day'
    if 'obs_date__ge' in raw_query_params.keys():
        from_date = datetime.strptime(raw_query_params['obs_date__ge'], '%Y/%m/%d')
    else:
        from_date = datetime(2000, 01, 01)
    from_date = truncate(from_date, agg)
    if 'obs_date__le' in raw_query_params.keys():
        to_date = datetime.strptime(raw_query_params['obs_date__le'], '%Y/%m/%d')
    else:
        to_date = datetime.now()
    to_date = truncate(to_date, agg)
    # Pull data from meta_table
    meta_table = Table('sf_meta', Base.metadata, autoload=True,
        autoload_with=engine)
    meta_query = session.query(
        meta_table.c['table_name'],
        meta_table.c['human_name'],
        meta_table.c['duration'],
    ).filter('dist_q')
    if dataset_name:
        meta_query = meta_query.filter(meta_table.c['table_name'] == dataset_name)
    datasets = meta_query.all()
    # Retrieve the census tracts inside the considered area (whose
    # centroids are inside the considered area)
    # A buffer is added to consider centroids that may be in the water
    land_geom = contour_intersect(query_geom, buffer_radius=0.0)
    #land_geom = query_geom
    blocks_raw_query_params['centroid__within'] = land_geom
    blocks_table = Table('sf_census_blocks', Base.metadata,
        autoload=True, autoload_with=engine)
    resp = {
        'meta': {
            'status': 'ok',
            'message': '',
        },
        'objects': [],
    }
    status_code = 200
    for dataset in datasets:
        table_name = dataset[0]
        human_name = dataset[1]
        duration = dataset[2]
        table = Table(table_name, Base.metadata,
            autoload=True, autoload_with=engine)
        local_params = raw_query_params.copy()
        if 'obs_date__ge' in local_params.keys() and duration == 'interval':
            local_params['end_date__ge'] = local_params['obs_date__ge']
            del local_params['obs_date__ge']
        if 'obs_date__le' in local_params.keys() and duration == 'interval':
            local_params['start_date__le'] = local_params['obs_date__le']
            del local_params['obs_date__le']
        valid_query, query_clauses, resp, status_code =\
            make_query(table, local_params, resp)
        valid_query, blocks_query_clauses, resp, status_code =\
            make_query(blocks_table, blocks_raw_query_params, resp)
        if valid_query:
            if duration == 'interval':
                # Retrieve the dates when "something changes"
                start_dates_q = session.query(table.c['start_date']\
                    .label('date'))\
                    .filter(table.c['start_date'] >=
                            local_params['end_date__ge'])\
                    .filter(table.c['start_date'] <=
                            local_params['start_date__le'])
                end_dates_q = session.query(table.c['end_date']\
                    .label('date'))\
                    .filter(table.c['end_date'] >=
                            local_params['end_date__ge'])\
                    .filter(table.c['end_date'] <=
                            local_params['start_date__le'])
                dates_query = start_dates_q.union(end_dates_q)
                dates = [from_date] + [v[0] for v in dates_query.all()]
                dates = sorted(dates)
                log = OrderedDict()
                for date in dates:
                    points_query = session.query(func.ST_Collect(table.c['geom'])\
                        .label('points'))\
                        .filter(table.c['start_date'] <= str(date))\
                        .filter(table.c['end_date'] > str(date))
                    points_query = points_query.subquery()
                    nested_query = session.query(
                        func.ST_Distance_Sphere(points_query.c['points'],
                                    blocks_table.c['centroid']).label('min_dist'),
                        blocks_table.c['pop10'].label('pop')
                    )
                    for clause in blocks_query_clauses:
                        nested_query = nested_query.filter(clause)
                    nested_query = nested_query.subquery()
                    # Compute the weighted average (by block population)
                    base_query = session.query(
                        func.sum(nested_query.c['min_dist'] * nested_query.c['pop']) /\
                            func.sum(nested_query.c['pop'])
                    )
                    value = base_query.first()[0] 
                    log[date] = value 
                # Replicate last value
                log[to_date] = value 
            else:
                table_obs_date = func.date_trunc(agg, table.c['obs_date'])
                log = OrderedDict()
                cursor = from_date
                while cursor <= to_date:
                    points_query = session.query(func.ST_Collect(table.c['geom'])\
                        .label('points'))\
                        .filter(table_obs_date == cursor)
                    points_query = points_query.subquery()
                    nested_query = session.query(
                        func.ST_Distance_Sphere(points_query.c['points'],
                                    blocks_table.c['centroid']).label('min_dist'),
                        blocks_table.c['pop10'].label('pop')
                    )
                    for clause in blocks_query_clauses:
                        nested_query = nested_query.filter(clause)
                    nested_query = nested_query.subquery()
                    # Compute the weighted average (by block population)
                    base_query = session.query(
                        func.sum(nested_query.c['min_dist'] * nested_query.c['pop']) /\
                            func.sum(nested_query.c['pop'])
                    )
                    value = base_query.first()[0]
                    log[cursor] = value if value else -1
                    cursor = increment_datetime(cursor, agg)
                #explain_query = session.execute(explain(base_query, analyze=True))
                #explain_values = [v for v in explain_query.fetchall()]
                
                #print '\nDIST_QUERY ({0}):'.format(human_name)
                #for v in explain_values:
                #    print v[0]
                #print '\n\n'
            d = {
                'dataset_name': table_name,
                'human_name': human_name,
                'query_type': 'dist',
                'response_type': 'time-series',
                'time_agg': agg
            }
            return_values = []
            for k in log:
                return_values.append({
                    'date': k,
                    'value': round(log[k], 4)
                })
            d['values'] = return_values
            resp['objects'].append(d)
        else:
            resp['meta']['status'] = 'error'
            resp['meta']['message'] = 'Invalid query.'
            resp['objects'] = []
            break
    return resp, status_code

def weighted():
    """
    This type of query uses datasets that provide information about some fixed
    geometries, like zip codes or census tracts.
    """
    raw_query_params = request.args.copy()
    dataset_name = None
    if 'dataset_name' in raw_query_params.keys():
        dataset_name = raw_query_params['dataset_name']
        del raw_query_params['dataset_name']
    del raw_query_params['obs_date__ge']
    del raw_query_params['obs_date__le']
    del raw_query_params['agg']
    if 'location_geom__within' in raw_query_params.keys():
        raw_query_params['geom__intersects'] = raw_query_params['location_geom__within']
        del raw_query_params['location_geom__within']
        val = json.loads(raw_query_params['geom__intersects'])['geometry']
        val['crs'] = {"type":"name", "properties":{"name":"EPSG:4326"}}
        query_geom = json.dumps(val)
    else:
        query_geom = None
    # Pull data from meta_table
    meta_table = Table('sf_meta', Base.metadata, autoload=True,
        autoload_with=engine)
    meta_query = session.query(
        meta_table.c['table_name'],
        meta_table.c['human_name'],
        meta_table.c['val_attr']
    ).filter('weighted_q')
    if dataset_name:
        meta_query = meta_query.filter(meta_table.c['table_name'] == dataset_name)
    datasets = meta_query.all()
    resp = {
        'meta': {
            'status': 'ok',
            'message': '',
        },
        'objects': [],
    }
    status_code = 200
    for dataset in datasets:
        table_name = dataset[0]
        human_name = dataset[1]
        val_attr = dataset[2]
        table = Table(table_name, Base.metadata,
            autoload=True, autoload_with=engine)
        valid_query, query_clauses, resp, status_code =\
            make_query(table, raw_query_params, resp)
        if valid_query:
            # Retrieve the shore contours to consider land only
            land_table = Table('sf_shore', Base.metadata,
                autoload=True, autoload_with=engine)
            # If a query geometry is provided, compute its intersection
            # with the shore contours; otherwise, just use the land
            # contours
            if query_geom:
                hot_geom = func.ST_Intersection(
                    func.ST_GeomFromGeoJSON(query_geom),
                    land_table.c['geom']
                )
            else:
                hot_geom = land_table.c['geom']
            land_val = session.query(func.ST_AsGeoJSON(hot_geom)).first()[0]
            land_val = json.loads(land_val)
            land_val['crs'] = {"type":"name","properties":{"name":"EPSG:4326"}}
            land_geom = json.dumps(land_val)
            if query_geom:
                # compute the intersections
                hot_geom = func.ST_Intersection(func.ST_GeomFromGeoJSON(land_geom),
                                                table.c['geom'])
            else:
                # if no query_geom is provided, just consider everything
                hot_geom = table.c['geom']
            base_query = session.query(
                func.sum(func.ST_Area(hot_geom) * table.c[val_attr]) /\
                    func.ST_Area(func.ST_GeomFromGeoJSON(land_geom))
            )
            # Applying this filtering makes the query compute the actual
            # intersection only with polygons that actually intersects
            for clause in query_clauses:
                base_query = base_query.filter(clause)
            values = [v for v in base_query.all()]
            for v in values:
                d = {
                    'dataset_name': table_name, 
                    'human_name': human_name,
                    'query_type': 'weighted',
                    'response_type': 'single-value',
                    'value': round(v[0] if v[0] else 0.0, 4)
                }
                resp['objects'].append(d)
        else:
            resp['meta']['status'] = 'error'
            resp['meta']['message'] = 'Invalid query.'
            resp['objects'] = []
            break
    return resp, status_code

@api.route('/api/pop/')
@crossdomain(origin="*")
def pop(subquery=False):
    census_table = Table('sf_census_blocks', Base.metadata,
        autoload=True, autoload_with=engine)
    raw_query_params = request.args.copy()
    valid_query, query_clauses, resp, status_code = make_query(census_table, raw_query_params)
    if valid_query:
        base_query = session.query(func.sum(census_table.c['pop10']),
            func.sum(census_table.c['housing10']))
        for clause in query_clauses:
            base_query = base_query.filter(clause)
        values = [v for v in base_query.all()]
        for v in values:
            d = {'pop': v[0], 'housing': v[1]}
            resp['objects'].append(d)
        resp['meta']['status'] = 'ok'
    if not subquery:
        resp = make_response(json.dumps(resp, default=dthandler), status_code)
        resp.headers['Content-Type'] = 'application/json'
    return resp

@api.route('/api/indicators/')
@crossdomain(origin="*")
def indicators():
    resp_all = {
        'meta': {
            'status': 'error',
            'message': '',
        },
        'objects': [],
    }
    timetot1 = time.time()
    for name, attr in query_types.items():
        time1 = time.time()
        resp, status_code = attr['func']()
        time2 = time.time()
        print '{0}: {1}'.format(name, time2-time1)
        time1 = time.time()
        for obj in resp['objects']:
            resp_all['objects'].append(obj)
        time2 = time.time()
        print '\tExtra: {0}'.format(time2-time1)
    timetot2 = time.time()
    print 'TOTAL: {0}'.format(timetot2-timetot1)
    resp_all['meta']['status'] = 'ok'
    resp_all = make_response(json.dumps(resp_all, default=dthandler), status_code)
    resp_all.headers['Content-Type'] = 'application/json'
    return resp_all


@api.route('/api/master/')
@crossdomain(origin="*")
def dataset():
    raw_query_params = request.args.copy()
    agg = raw_query_params.get('agg')
    if not agg:
        agg = 'day'
    else:
        del raw_query_params['agg']
    datatype = 'json'
    if raw_query_params.get('datatype'):
        datatype = raw_query_params['datatype']
        del raw_query_params['datatype']
    valid_query, query_clauses, resp, status_code = make_query(MasterTable,raw_query_params)
    if valid_query:
        time_agg = func.date_trunc(agg, MasterTable.c['obs_date'])
        base_query = session.query(time_agg, 
            func.count(MasterTable.c['obs_date']),
            MasterTable.c['dataset_name'])
        base_query = base_query.filter(MasterTable.c['current_flag'] == True)
        for clause in query_clauses:
            base_query = base_query.filter(clause)
        base_query = base_query.group_by(MasterTable.c['dataset_name'])\
            .group_by(time_agg)\
            .order_by(time_agg)
        values = [o for o in base_query.all()]
        results = []
        for value in values:
            d = {
                'dataset_name': value[2],
                'group': value[0],
                'count': value[1],
                }
            results.append(d)
        results = sorted(results, key=itemgetter('dataset_name'))
        for k,g in groupby(results, key=itemgetter('dataset_name')):
            d = {'dataset_name': k}
            d['temporal_aggregate'] = agg
            d['items'] = list(g)
            resp['objects'].append(d)
        resp['meta']['status'] = 'ok'
    if datatype == 'json':
        resp = make_response(json.dumps(resp, default=dthandler), status_code)
        resp.headers['Content-Type'] = 'application/json'
    elif datatype == 'csv':
        csv_resp = []
        fields = ['temporal_group']
        results = sorted(results, key=itemgetter('group'))
        for k,g in groupby(results, key=itemgetter('group')):
            d = [k]
            for row in list(g):
                if row['dataset_name'] not in fields:
                    fields.append(row['dataset_name'])
                d.append(row['count'])
            csv_resp.append(d)
        csv_resp[0] = fields
        csv_resp = make_csv(csv_resp)
        resp = make_response(csv_resp, 200)
        resp.headers['Content-Type'] = 'text/csv'
        filedate = datetime.now().strftime('%Y-%m-%d')
        resp.headers['Content-Disposition'] = 'attachment; filename=%s.csv' % (filedate)
    return resp

def parse_join_query(params):
    queries = {
        'base' : {},
        'detail': {},
    }
    agg = 'day'
    datatype = 'json'
    for key, value in params.items():
        if key.split('__')[0] in ['obs_date', 'location_geom', 'dataset_name']:
            queries['base'][key] = value
        elif key == 'agg':
            agg = value
        elif key == 'datatype':
            datatype = value
        else:
            queries['detail'][key] = value
    return agg, datatype, queries

@api.route('/api/detail/')
@crossdomain(origin="*")
def detail():
    raw_query_params = request.args.copy()
    agg, datatype, queries = parse_join_query(raw_query_params)
    limit = raw_query_params.get('limit')
    order_by = raw_query_params.get('order_by')
    valid_query, base_clauses, resp, status_code = make_query(MasterTable, queries['base'])
    if valid_query:
        resp['meta']['status'] = 'ok'
        dname = raw_query_params['dataset_name']
        dataset = Table('dat_%s' % dname, Base.metadata,
            autoload=True, autoload_with=engine,
            extend_existing=True)
        base_query = session.query(MasterTable.c.obs_date, dataset)
        valid_query, detail_clauses, resp, status_code = make_query(dataset, queries['detail'])
        if valid_query:
            resp['meta']['status'] = 'ok'
            pk = [p.name for p in dataset.primary_key][0]
            base_query = base_query.join(dataset, MasterTable.c.dataset_row_id == dataset.c[pk])
        for clause in base_clauses:
            base_query = base_query.filter(clause)
        if order_by:
            col, order = order_by.split(',')
            base_query = base_query.order_by(getattr(MasterTable.c[col], order)())
        for clause in detail_clauses:
            base_query = base_query.filter(clause)
        if limit:
            base_query = base_query.limit(limit)
        values = [r for r in base_query.all()]
        fieldnames = dataset.columns.keys()
        for value in values:
            d = {}
            for k,v in zip(fieldnames, value[1:]):
                d[k] = v
            resp['objects'].append(d)
        resp['meta']['total'] = len(resp['objects'])
    if datatype == 'json':
        resp = make_response(json.dumps(resp, default=dthandler), status_code)
        resp.headers['Content-Type'] = 'application/json'
    elif datatype == 'csv':
        csv_resp = [fieldnames]
        csv_resp.extend([v[1:] for v in values])
        resp = make_response(make_csv(csv_resp), 200)
        filedate = datetime.now().strftime('%Y-%m-%d')
        dname = raw_query_params['dataset_name']
        filedate = datetime.now().strftime('%Y-%m-%d')
        resp.headers['Content-Type'] = 'text/csv'
        resp.headers['Content-Disposition'] = 'attachment; filename=%s_%s.csv' % (dname, filedate)
    return resp

@api.route('/api/detail-aggregate/')
@crossdomain(origin="*")
def detail_aggregate():
    raw_query_params = request.args.copy()
    agg, datatype, queries = parse_join_query(raw_query_params)
    valid_query, base_clauses, resp, status_code = make_query(MasterTable, queries['base'])
    if valid_query:
        resp['meta']['status'] = 'ok'
        time_agg = func.date_trunc(agg, MasterTable.c['obs_date'])
        base_query = session.query(time_agg, func.count(MasterTable.c.dataset_row_id))
        dname = raw_query_params['dataset_name']
        dataset = Table('dat_%s' % dname, Base.metadata,
            autoload=True, autoload_with=engine,
            extend_existing=True)
        valid_query, detail_clauses, resp, status_code = make_query(dataset, queries['detail'])
        if valid_query:
            resp['meta']['status'] = 'ok'
            pk = [p.name for p in dataset.primary_key][0]
            base_query = base_query.join(dataset, MasterTable.c.dataset_row_id == dataset.c[pk])
            for clause in base_clauses:
                base_query = base_query.filter(clause)
            for clause in detail_clauses:
                base_query = base_query.filter(clause)
            values = [r for r in base_query.group_by(time_agg).order_by(time_agg).all()]
            items = []
            for value in values:
                d = {
                    'group': value[0],
                    'count': value[1]
                }
                items.append(d)
            resp['objects'].append({
                'temporal_aggregate': agg,
                'dataset_name': ' '.join(dname.split('_')).title(),
                'items': items
            })
    resp = make_response(json.dumps(resp, default=dthandler), status_code)
    resp.headers['Content-Type'] = 'application/json'
    return resp

def getSizeInDegrees(meters, latitude):
    size_x = abs(meters / (111111.0 * math.cos(latitude)))
    size_y = meters / 111111.0
    return size_x, size_y

@api.route('/api/grid/')
@crossdomain(origin="*")
def grid():
    dataset_name = request.args.get('dataset_name')
    resolution = request.args.get('resolution')
    obs_to = request.args.get('obs_date__le')
    obs_from = request.args.get('obs_date__ge')
    location_geom = request.args.get('location_geom__within')
    buff = request.args.get('buffer', 100)
    center = request.args.getlist('center[]')
    resp = {'type': 'FeatureCollection', 'features': []}
    size_x, size_y = getSizeInDegrees(float(resolution), float(center[0]))
    if location_geom:
        location_geom = json.loads(location_geom)['geometry']
        if location_geom['type'] == 'LineString':
            shape = asShape(location_geom)
            lat = shape.centroid.y
            # 100 meters by default
            x, y = getSizeInDegrees(int(buff), lat)
            size_x, size_y = getSizeInDegrees(50, lat)
            location_geom = shape.buffer(y).__geo_interface__
        location_geom['crs'] = {"type":"name","properties":{"name":"EPSG:4326"}}
    query = session.query(func.count(MasterTable.c.dataset_row_id), 
            func.ST_SnapToGrid(MasterTable.c.location_geom, size_x, size_y))\
            .filter(MasterTable.c.dataset_name == dataset_name)
    if obs_from:
        query = query.filter(MasterTable.c.obs_date >= obs_from)
    if obs_to:
        query = query.filter(MasterTable.c.obs_date <= obs_to)
    if location_geom:
        query = query.filter(MasterTable.c.location_geom\
                .ST_Within(func.ST_GeomFromGeoJSON(json.dumps(location_geom))))
    query = query.group_by(func.ST_SnapToGrid(MasterTable.c.location_geom, size_x, size_y))
    values = [d for d in query.all()]
    for value in values:
        d = {
            'type': 'Feature', 
            'properties': {
                'count': value[0], 
            },
        }
        if value[1]:
            pt = loads(value[1].decode('hex'))
            south, west = (pt.x - (size_x / 2)), (pt.y - (size_y /2))
            north, east = (pt.x + (size_x / 2)), (pt.y + (size_y / 2))
            d['geometry'] = box(south, west, north, east).__geo_interface__
        resp['features'].append(d)
    resp = make_response(json.dumps(resp, default=dthandler))
    resp.headers['Content-Type'] = 'application/json'
    return resp

