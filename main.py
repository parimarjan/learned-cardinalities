from cardinality_estimation import *
import argparse
import park
from park.param import parser
import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def main():
    print("starting main")
    env = park.make("query_optimizer")
    # con = connect(user='****', host = 'localhost', password='****')
    con = pg.connect(user="imdb", host="localhost", port="5400",
            password="imdb")
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = con.cursor()
    cur.execute('CREATE DATABASE ' + "test")
    cur.close()
    con.close()

def read_flags():
    # parser = argparse.ArgumentParser()
    return parser.parse_args()

args = read_flags()
main()
