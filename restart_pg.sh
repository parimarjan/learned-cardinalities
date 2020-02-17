#!/usr/bin/env bash
echo "drop cache!"
pg_ctl -D $PG_DATA_DIR -m i restart -l logfile
#sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
#sudo systemctl restart postgresql
