import os
import re
from flask import request
import psycopg2
from sqlalchemy import create_engine, types
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from psycopg2.extensions import adapt, register_adapter, AsIs

# Read the connection parameters from file
#if 'WOPR_CONN' not in os.environ.keys():
#    os.environ['WOPR_CONN'] = open('/home/ubuntu/wopr_conn', 'r').readline()

app_engine = create_engine(os.environ['WOPR_CONN'], convert_unicode=True)
task_engine = create_engine(
    os.environ['WOPR_CONN'], 
    convert_unicode=True,
    poolclass=NullPool)
session = scoped_session(sessionmaker(bind=app_engine,
                                      autocommit=False,
                                      autoflush=False))
Base = declarative_base()
Base.query = session.query_property()

class Point(types.UserDefinedType):
    
    def get_col_spec(self):
        return 'POINT'

    def bind_processor(dialect):
        def process(value):
            return '(%s)' % ','.join([v for v in value])
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            m = re.match(r"\(([^)]+),([^)]+)\)", value)
            if m:
                return (float(m.group(1)), float(m.group(2)))
            else:
                raise psycopg2.InterfaceError("bad point representation: %r" % value)
        return process
