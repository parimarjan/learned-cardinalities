#!/usr/bin/env bash
echo "drop cache!"
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
echo "drop cache done!"
#sudo systemctl restart postgresql
/pgfs/mysql-server/debug/bin/mysqladmin -u root shutdown
/pgfs/mysql-server/debug/bin/mysqld -u root &
