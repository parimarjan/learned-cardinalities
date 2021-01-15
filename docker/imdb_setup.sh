#!/bin/sh

createdb -U arthurfleck imdb

wget -O /var/lib/postgresql/pg_imdb.tar cs.brandeis.edu/~rcmarcus/pg_imdb.tar
tar xfv /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/
psql -d imdb -U arthurfleck -c "SHOW max_wal_size";
pg_restore -v -d imdb -U arthurfleck /var/lib/postgresql/pg_imdb
