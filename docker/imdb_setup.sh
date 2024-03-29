#!/bin/sh

#createdb -U arthurfleck imdb
createdb -U $POSTGRES_USER imdb

wget -O /var/lib/postgresql/pg_imdb.tar cs.brandeis.edu/~rcmarcus/pg_imdb.tar
tar xfv /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/
#psql -d imdb -U arthurfleck -c "SHOW max_wal_size";
psql -d imdb -U $POSTGRES_USER -c "SHOW max_wal_size";
#pg_restore -v -d imdb -U arthurfleck /var/lib/postgresql/pg_imdb
pg_restore -v -d imdb -U $POSTGRES_USER /var/lib/postgresql/pg_imdb
