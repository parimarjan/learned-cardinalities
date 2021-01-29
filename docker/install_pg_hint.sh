#!/bin/sh
git clone https://github.com/parimarjan/pg_hint_plan.git
cd pg_hint_plan
make
make install
#sed -i 's/max_wal_size = 1GB/max_wal_size = 20GB/g' /var/lib/postgresql/data/postgresql.conf
