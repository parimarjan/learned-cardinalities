SELECT COUNT(*)
FROM
osm2
WHERE 'Xc0' < osm2.c0
AND osm2.c0 <= 'Yc0'
AND 'Xc2' < osm2.c2
AND osm2.c2 <= 'Yc2'
AND osm2.d0 IN ('SELECT osm2.d0 FROM osm2 WHERE osm2.d0 IS NOT NULL')
AND osm2.d1 IN ('SELECT osm2.d1 FROM osm2 WHERE osm2.d1 IS NOT NULL')
