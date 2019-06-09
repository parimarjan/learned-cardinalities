SELECT COUNT(*)
FROM title as t,
movie_keyword as mk,
keyword as k,
info_type as it,
movie_info as mi
WHERE it.id = mi.info_type_id
AND mi.movie_id = t.id
AND mk.keyword_id = k.id
AND mk.movie_id = t.id
AND k.keyword = 'keyword'
AND mi.info = 'info'
AND it.id = 'id'
