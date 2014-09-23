import requests
import os
from datetime import datetime, date
from sqlalchemy import Column, Integer, Table, func, select, Boolean, \
    Date, DateTime, UniqueConstraint, text, and_, or_
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.dialects.postgresql import TIMESTAMP
from geoalchemy2 import Geometry
from wopr.database import task_engine as engine, Base
from wopr.models import crime_table, MasterTable, shp2table
import gzip
from zipfile import ZipFile
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon
import json
import pyproj
import urllib2

CRIMES = 'http://apps.sfgov.org/datafiles/view.php?file=Police/sfpd_incident_all_csv.zip'
DATA_DIR = os.environ['WOPR_DATA_DIR']

def cleanup_temp_tables():
    tables = ['new', 'src', 'raw', 'chg', 'dedup']
    for table in tables:
        try:
            t = Table('%s_chicago_crimes_all' % table, Base.metadata, 
                autoload=True, autoload_with=engine, extend_existing=True)
            t.drop(bind=engine, checkfirst=True)
        except NoSuchTableError:
            pass
    return 'Temp tables dropped'

def download_crime():
    request = urllib2.urlopen(CRIMES)
    fpath = '{0}/sf_crime_{1}.zip'\
                .format(DATA_DIR, datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    output = open(fpath, 'w')
    output.write(request.read())
    output.close()
    return fpath

def dat_crime():
    # Step Zero: Create dat_crime table
    try:
        src_crime_table = Table('src_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        src_crime_table = src_crime()
    dat_crime_table = crime_table('dat_chicago_crimes_all', Base.metadata)
    dat_crime_table.append_column(Column('chicago_crimes_all_row_id', Integer, primary_key=True))
    dat_crime_table.append_column(Column('start_date', TIMESTAMP, default=datetime.now))
    dat_crime_table.append_column(Column('end_date', TIMESTAMP, default=None))
    dat_crime_table.append_column(Column('current_flag', Boolean, default=True))
    dat_crime_table.append_constraint(UniqueConstraint('id', 'start_date'))
    dat_crime_table.create(bind=engine, checkfirst=True)
    new_cols = ['start_date', 'end_date', 'current_flag', 'chicago_crimes_all_row_id']
    dat_ins = dat_crime_table.insert()\
        .from_select(
            [c for c in dat_crime_table.columns.keys() if c not in new_cols],
            select([c for c in src_crime_table.columns])
        )
    conn = engine.connect()
    conn.execute(dat_ins)
    return dat_crime_table

def raw_crime(fpath=None, tablename='raw_chicago_crimes_all'):
    # Step One: Load raw downloaded data
    if not fpath:
        fpath = download_crime()
    raw_crime_table = crime_table(tablename, Base.metadata)
    raw_crime_table.drop(bind=engine, checkfirst=True)
    raw_crime_table.append_column(Column('dup_row_id', Integer, primary_key=True))
    raw_crime_table.create(bind=engine)
    conn = engine.raw_connection()
    cursor = conn.cursor()
    with gzip.open(fpath, 'rb') as f:
        cursor.copy_expert("COPY %s \
            (id, case_number, orig_date, block, iucr, primary_type, \
            description, location_description, arrest, domestic, \
            beat, district, ward, community_area, fbi_code, \
            x_coordinate, y_coordinate, year, updated_on, \
            latitude, longitude, location) FROM STDIN WITH \
            (FORMAT CSV, HEADER true, DELIMITER ',')" % tablename, f)
    conn.commit()
    return raw_crime_table

def dedupe_crime():
    # Step Two: Find duplicate records by case_number
    try:
        raw_crime_table = Table('raw_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        raw_crime_table = raw_crime(tablename='raw_chicago_crimes_all')
    dedupe_crime_table = Table('dedup_chicago_crimes_all', Base.metadata,
        Column('dup_row_id', Integer, primary_key=True),
        extend_existing=True)
    dedupe_crime_table.drop(bind=engine, checkfirst=True)
    dedupe_crime_table.create(bind=engine)
    ins = dedupe_crime_table.insert()\
        .from_select(
            ['dup_row_id'], 
            select([func.max(raw_crime_table.c.dup_row_id)])\
            .group_by(raw_crime_table.c.id)
        )
    conn = engine.connect()
    conn.execute(ins)
    return dedupe_crime_table

def src_crime():
    # Step Three: Create New table with unique ids
    try:
        raw_crime_table = Table('raw_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        raw_crime_table = raw_crime()
    try:
        dedupe_crime_table = Table('dedup_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dedupe_crime_table = dedupe_crime()
    src_crime_table = crime_table('src_chicago_crimes_all', Base.metadata)
    src_crime_table.drop(bind=engine, checkfirst=True)
    src_crime_table.create(bind=engine)
    ins = src_crime_table.insert()\
        .from_select(
            src_crime_table.columns.keys(),
            select([c for c in raw_crime_table.columns if c.name != 'dup_row_id'])\
                .where(raw_crime_table.c.dup_row_id == dedupe_crime_table.c.dup_row_id)
        )
    conn = engine.connect()
    conn.execute(ins)
    return src_crime_table

def new_crime():
    # Step Four: Find New Crimes
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    try:
        src_crime_table = Table('src_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        src_crime_table = src_crime()
    new_crime_table = Table('new_chicago_crimes_all', Base.metadata, 
        Column('id', Integer, primary_key=True),
        extend_existing=True)
    new_crime_table.drop(bind=engine, checkfirst=True)
    new_crime_table.create(bind=engine)
    ins = new_crime_table.insert()\
        .from_select(
            ['id'],
            select([src_crime_table.c.id])\
                .select_from(src_crime_table.join(dat_crime_table, 
                    src_crime_table.c.id == dat_crime_table.c.id, isouter=True))\
                .where(dat_crime_table.c.chicago_crimes_all_row_id == None)
        )
    conn = engine.connect()
    try:
        conn.execute(ins)
        return new_crime_table
    except TypeError:
        # No new records
        return None

def update_dat_crimes():
    # Step Five: Update Main Crime table
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    try:
        src_crime_table = Table('src_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        src_crime_table = src_crime()
    try:
        new_crime_table = Table('new_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        new_crime_table = new_crime()
    excluded_cols = ['end_date', 'current_flag', 'chicago_crimes_all_row_id']
    dat_cols = [c for c in dat_crime_table.columns.keys() if c not in excluded_cols]
    excluded_cols.append('start_date')
    src_cols = [c for c in src_crime_table.columns if c.name not in excluded_cols]
    src_cols.append(text("'%s' AS start_date" % datetime.now().strftime('%Y-%m-%d')))
    ins = dat_crime_table.insert()\
        .from_select(
            dat_cols,
            select(src_cols)\
                .select_from(src_crime_table.join(new_crime_table,
                    src_crime_table.c.id == new_crime_table.c.id))
        )
    conn = engine.connect()
    conn.execute(ins)
    return 'Crime Table updated'

def update_master():
    # Step Six: Update Master table
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    try:
        new_crime_table = Table('new_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        new_crime_table = new_crime()
    col_names = ['start_date', 'end_date', 'current_flag', 'location', 'latitude', 'longitude']
    cols = [
        dat_crime_table.c.start_date,
        dat_crime_table.c.end_date,
        dat_crime_table.c.current_flag,
        dat_crime_table.c.location,
        dat_crime_table.c.latitude, 
        dat_crime_table.c.longitude,
    ]
    cols.append(dat_crime_table.c.orig_date.label('obs_date'))
    cols.append(text("NULL AS obs_ts"))
    cols.append(text("NULL AS geotag1"))
    cols.append(text("NULL AS geotag2"))
    cols.append(text("NULL AS geotag3"))
    cols.append(text("'chicago_crimes_all' AS dataset_name"))
    cols.append(dat_crime_table.c.chicago_crimes_all_row_id.label('dataset_row_id'))
    cols.append(text("ST_PointFromText('POINT(' || dat_chicago_crimes_all.longitude || ' ' || dat_chicago_crimes_all.latitude || ')', 4326) as location_geom"))
    ins = MasterTable.insert()\
        .from_select(
            [c for c in MasterTable.columns.keys() if c != 'master_row_id'],
            select(cols)\
                .select_from(dat_crime_table.join(new_crime_table, 
                    dat_crime_table.c.id == new_crime_table.c.id))
        )
    conn = engine.connect()
    conn.execute(ins)
    return 'Master updated'

def chg_crime():
    # Step Seven: Find updates
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    try:
        src_crime_table = Table('src_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        src_crime_table = new_crime()
    chg_crime_table = Table('chg_chicago_crimes_all', Base.metadata, 
        Column('id', Integer, primary_key=True),
        extend_existing=True)
    chg_crime_table.drop(bind=engine, checkfirst=True)
    chg_crime_table.create(bind=engine)
    src_cols = [c for c in src_crime_table.columns if c.name != 'id']
    dat_cols = [c for c in dat_crime_table.columns if c.name != 'id']
    and_args = []
    for s, d in zip(src_cols, dat_cols):
        ors = or_(s != None, d != None)
        ands = and_(ors, s != d)
        and_args.append(ands)
    ins = chg_crime_table.insert()\
          .from_select(
              ['id'],
              select([src_crime_table.c.id])\
                  .select_from(src_crime_table.join(dat_crime_table,
                      src_crime_table.c.id == dat_crime_table.c.id))\
                  .where(or_(
                          and_(dat_crime_table.c.current_flag == True, 
                                and_(or_(src_crime_table.c.id != None, dat_crime_table.c.id != None), 
                                src_crime_table.c.id != dat_crime_table.c.id)),
                          *and_args))
          )
    conn = engine.connect()
    conn.execute(ins)
    return chg_crime_table

def update_crime_current_flag():
    # Step Seven: Update end_date and current_flag in crime table
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    try:
        chg_crime_table = Table('chg_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        chg_crime_table = chg_crime()
    update = dat_crime_table.update()\
        .values(current_flag=False, end_date=datetime.now().strftime('%Y-%m-%d'))\
        .where(dat_crime_table.c.id==chg_crime_table.c.id)\
        .where(dat_crime_table.c.current_flag == True)
    print update
    conn = engine.connect()
    conn.execute(update)
    return 'Crime table current flag updated'

def update_master_current_flag():
    # Step Eight: Update end_date and current_flag in master table
    try:
        dat_crime_table = Table('dat_chicago_crimes_all', Base.metadata, 
            autoload=True, autoload_with=engine, extend_existing=True)
    except NoSuchTableError:
        dat_crime_table = dat_crime()
    update = MasterTable.update()\
        .values(current_flag=False, end_date=datetime.now().strftime('%Y-%m-%d'))\
        .where(MasterTable.c.dataset_row_id == dat_crime_table.c.chicago_crimes_all_row_id)\
        .where(dat_crime_table.c.current_flag==False)\
        .where(dat_crime_table.c.end_date==date.today())
    conn = engine.connect()
    conn.execute(update)
    return 'Master table current flag updated'

