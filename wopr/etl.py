import os
from datetime import datetime, timedelta
from wopr.database import task_engine as engine, Base
from wopr.models import crime_table, MasterTable, sf_crime_table,\
    sf_meta_table, shp2table
from wopr.helpers import download_crime
from datetime import datetime, date
from sqlalchemy import Column, Integer, Table, func, select, Boolean,\
    UniqueConstraint, text, and_, or_
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement
from geoalchemy2.shape import from_shape
import gzip
from raven.handlers.logging import SentryHandler
from raven.conf import setup_logging
from zipfile import ZipFile
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon
import json
import pyproj

def sf_dat_crime(fpath=None, crime_type='violent'):
    #raw_crime = sf_raw_crime(fpath=fpath)
    # Assume for now there's no duplicate in the raw data, which means we don't
    # - dedupe_crime()
    # - and don't create src_crime()
    raw_crime_table = Table('raw_sf_crimes_all', Base.metadata,
        autoload=True, autoload_with=engine, extend_existing=True)
    if crime_type == 'violent':
        categories = ['ASSAULT', 'ROBBERY', 'SEX OFFENSES, FORCIBLE']
    elif crime_type == 'property':
        categories = ['LARCENY/THEFT', 'VEHICLE THEFT', 'BURGLARY', 'STOLEN PROPERTY',\
                      'ARSON', 'VANDALISM']
    # Create table "dat_sf_crimes_all", that contains additional fields needed
    # by Plenario, in addition to the raw data
    dat_crime_table = sf_crime_table('dat_sf_crimes_{0}'.format(crime_type), Base.metadata)
    dat_crime_table.append_column(
        Column( 'sf_crimes_all_row_id', Integer,    primary_key=True                         ) )
    dat_crime_table.append_column(
        Column( 'start_date',           TIMESTAMP,  server_default=text('CURRENT_TIMESTAMP') ) )
    dat_crime_table.append_column(
        Column( 'end_date',             TIMESTAMP,  server_default=text('NULL')              ) )
    dat_crime_table.append_column(
        Column( 'current_flag',         Boolean,    server_default=text('TRUE')              ) )
    # Constrain (id, start_date) to be unique (?)
    # dat_crime_table.append_constraint(UniqueConstraint('id', 'start_date'))
    dat_crime_table.create(bind=engine, checkfirst=True)
    new_cols = ['start_date', 'end_date', 'current_flag', 'sf_crimes_all_row_id']
    # Insert data from raw_crime_table (to be src_crime_table when we'll check
    # for duplicates)
    dat_ins = dat_crime_table.insert()\
        .from_select(
            [c for c in dat_crime_table.columns.keys() if c not in new_cols],
            select([c for c in raw_crime_table.columns if c.name != 'dup_row_id'])\
                .where(raw_crime_table.c.category.in_(categories))
        )
    conn = engine.contextual_connect()
    res = conn.execute(dat_ins)
    cols = sf_crime_master_cols(dat_crime_table, crime_type=crime_type)
    master_ins = MasterTable.insert()\
        .from_select(
            [c for c in MasterTable.columns.keys() if c != 'master_row_id'],
            select(cols).select_from(dat_crime_table)
        )
    conn = engine.contextual_connect()
    res = conn.execute(master_ins)
    return 'DAT crime created'

def sf_raw_crime(fpath=None, tablename='raw_sf_crimes_all'):
    if not fpath:
        fpath = download_crime()
    print 'SF crime data downloaded\n\n'
    raw_crime_table = sf_crime_table(tablename, Base.metadata)
    raw_crime_table.drop(bind=engine, checkfirst=True)
    raw_crime_table.append_column(Column('dup_row_id', Integer, primary_key=True))
    raw_crime_table.create(bind=engine)
    conn = engine.raw_connection()
    cursor = conn.cursor()
    zf = ZipFile(fpath)
    # SF crime data has one file for each year...
    for fn in zf.namelist():
        with zf.open(fn, 'r') as f:
            cursor.copy_expert("COPY %s \
                (id, category, description, day_of_week, date, time, pd_district, \
                 resolution, location_str, longitude, latitude) FROM STDIN WITH \
                (FORMAT CSV, HEADER true, DELIMITER ',')" % tablename, f)
        print '{0} imported'.format(fn)
    conn.commit()
    zf.close()
    return 'Raw Crime data inserted'

def transform_proj(geom, source, target=4326):
    """Transform a geometry's projection.

    Keyword arguments:
    geom -- a (nested) list of points (i.e. geojson coordinates)
    source/target -- integer ESPG codes, or Proj4 strings
    """
    s_str = '+init=EPSG:{0}'.format(source) if type(source)==int else source
    t_str = '+init=EPSG:{0}'.format(target) if type(target)==int else target
    ps = pyproj.Proj(s_str, preserve_units=True)
    pt = pyproj.Proj(t_str, preserve_units=True)
    # This function works as a depth-first search, recursively calling itself until a
    # point is found, and converted (base case)
    if type(geom[0]) == list:
        res = []
        for r in geom:
            res.append(transform_proj(r, source, target))
        return res
    else: # geom must be a point
        res = pyproj.transform(ps, pt, geom[0], geom[1])
        return list(res)
    
def import_shapefile(fpath, name, force_multipoly=False, proj=4326):
    """Import a shapefile into the PostGIS database

    Keyword arguments:
    fpath -- path to a zipfile to be extracted
    name -- name given to the newly created table
    force_multipoly -- enforce that the gemoetries are multipolygons
    proj -- source projection spec (EPSG code or Proj$ string)
    """
    # Open the shapefile with fiona.
    with fiona.open('/', vfs='zip://{0}'.format(fpath)) as shp:
        shp_table = shp2table(name, Base.metadata, shp.schema,
            force_multipoly=force_multipoly)
        shp_table.drop(bind=engine, checkfirst=True)
        shp_table.append_column(Column('row_id', Integer, primary_key=True))
        # If the geometry is not "point", append a centroid column
        if shp.schema['geometry'].lower() != 'point':
            shp_table.append_column(Column('centroid', Geometry('POINT',
                                    srid=4326)))
        shp_table.create(bind=engine)
        features = []
        count = 0
        num_shapes = len(shp)
        for r in shp:
            # ESRI shapefile don't contemplate multipolygons, i.e. the geometry
            # type is polygon even if multipolygons are contained.
            # If and when the 1st multipoly is encountered, the table is
            # re-initialized.
            if not force_multipoly and r['geometry']['type'] == 'MultiPolygon':
                return import_shapefile(fpath, name, force_multipoly=True, proj=proj)
            row_dict = dict((k.lower(), v) for k, v in r['properties'].iteritems())
            # GeoJSON intermediate representation
            geom_json = json.loads(str(r['geometry']).replace('\'', '"')\
                                   .replace('(', '[').replace(')', ']'))
            # If the projection is not long/lat (WGS84 - EPGS:4326), transform.
            if proj != 4326:
                geom_json['coordinates'] = transform_proj(geom_json['coordinates'], proj, 4326)
            # Shapely intermediate representation, used to obtained the WKT
            geom = shape(geom_json)
            if force_multipoly and r['geometry']['type'] != 'MultiPolygon':
                geom = MultiPolygon([geom])
            row_dict['geom'] = 'SRID=4326;{0}'.format(geom.wkt)
            if shp.schema['geometry'].lower() != 'point':
                row_dict['centroid'] =\
                    'SRID=4326;{0}'.format(geom.centroid.wkt)
            features.append(row_dict)
            count += 1
            #if count > 100: break
            # Buffer DB writes
            if not count % 1000 or count == num_shapes:
                try:
                    ins = shp_table.insert(features)
                    conn = engine.contextual_connect()
                    conn.execute(ins)
                except SQLAlchemyError as e:
                    print type(e)
                    print e.orig
                    return "Failed."
                features = []
                print count
    return 'Table {0} created from shapefile'.format(name)

def create_meta_table():
    """ Create the meta table containing information about the different
    datasets """
    table = sf_meta_table(Base.metadata)
    table.drop(bind=engine, checkfirst=True)
    table.create(bind=engine)
    return "Meta table created."

def add_dataset_meta(name, file_name='', human_name='', description='',
    count_q=False, area_q=False, dist_q=False, temp_q=False):
    """ Add infotmation about a dataset in the meta table """
    if human_name == '':
        human_name = name
    meta_table = Table('sf_meta', Base.metadata,
        autoload=True, autoload_with=engine, extend_existing=True)
    row = {'table_name': name, 'file_name': file_name,
        'human_name': human_name,'description': description,
        'last_update': func.current_timestamp(), 'count_q': count_q,
        'area_q': area_q, 'dist_q': dist_q, 'temp_q': temp_q}
    ins = meta_table.insert(row)
    conn = engine.contextual_connect()
    conn.execute(ins)
    return 'Meta information for {0} inserted.'.format(name)
