SELECT COUNT(*)
FROM
osm2
WHERE 'Xc0' < osm2.c0
AND osm2.c0 <= 'Yc0'
AND 'Xc1' < osm2.c1
AND osm2.c1 <= 'Yc1'
AND 'Xc2' < osm2.c2
AND osm2.c2 <= 'Yc2'
AND osm2.d0 IN (Xd0)
AND osm2.d1 IN (Xd1)
