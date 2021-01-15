#FROM postgres:9.4
FROM postgres:12

RUN apt-get update && apt-get install -y \
    wget \
    tar \
    git \
    gcc \
    build-essential \
    postgresql-server-dev-12

#ENV POSTGRES_USER="$LCARD_USER"
ENV POSTGRES_USER=arthurfleck
ENV POSTGRES_PASSWORD=password
# scripts in this directory will be run by the postgres docker init script

#COPY ./_test.sh /docker-entrypoint-initdb.d/
COPY ./init_db.sh /docker-entrypoint-initdb.d/_init_db.sh

#RUN sed -i 's/max_wal_size = 1GB/max_wal_size = 50GB/g' /var/lib/postgresql/data/postgresql.conf

USER root
ADD install_pg_hint.sh /
RUN chmod +x /install_pg_hint.sh
RUN /install_pg_hint.sh

ADD imdb_setup.sh /
RUN chmod +x /imdb_setup.sh
# not running it right away because we want to restart docker container so
# postgresql conf variables take effect

