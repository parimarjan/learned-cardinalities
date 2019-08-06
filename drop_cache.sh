#!/usr/bin/env bash
echo "drop cache!"
pg_ctl -D $PG_DATA_DIR -l logfile restart
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
echo "drop cache done!"
