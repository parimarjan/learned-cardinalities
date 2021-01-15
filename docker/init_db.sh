#!/bin/sh
#createdb -U "$CARD_USER" imdb

sed -i 's/max_wal_size = 1GB/max_wal_size = 50GB/g' /var/lib/postgresql/data/postgresql.conf
echo "done updating conf file"

# to init imdb DB
#wget -O /var/lib/postgresql/pg_imdb.tar cs.brandeis.edu/~rcmarcus/pg_imdb.tar
#tar xfv /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/
# restart db?
#pg_restore -v -d imdb -U arthurfleck /var/lib/postgresql/pg_imdb

