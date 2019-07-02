echo "starting to download imdb database"
# need to be careful of copying files into a directory which is not owned by
# root, since these scripts will be executed as user postgres by the official
# docker postgres init script

# to init imdb DB
wget -O /var/lib/postgresql/pg_imdb.tar cs.brandeis.edu/~rcmarcus/pg_imdb.tar
tar xfv /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/
pg_restore -v -d imdb -U imdb /var/lib/postgresql/pg_imdb
