#!/usr/bin/env bash
echo "drop cache!"
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
echo "drop cache done!"
#sudo systemctl restart postgresql
