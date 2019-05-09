# pg_restore for the imdb db we have doesn't seem to work with the latest
# postgres
FROM postgres:9.4
RUN apt-get update
RUN apt-get install -y wget tar
ENV POSTGRES_USER=card_est
ENV POSTGRES_PASSWORD=card_est
# scripts in this directory will be run by the postgres docker init script
COPY ./init_db.sh /docker-entrypoint-initdb.d/
