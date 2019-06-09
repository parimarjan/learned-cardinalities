SELECT it.id, mi.id
FROM info_type as it, movie_info as mi
WHERE it.id = mi.info_type_id
AND it.id = 'id'
AND mi.info = 'info'
