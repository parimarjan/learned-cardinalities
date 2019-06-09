SELECT COUNT(*)
FROM
movie_keyword,
title,
keyword,
movie_info,
info_type,
kind_type
WHERE
title.id = movie_keyword.movie_id
AND movie_keyword.keyword_id = keyword.id
AND movie_info.movie_id = title.id
AND movie_info.info_type_id = info_type.id
AND title.kind_id = kind_type.id
AND info_type.id = '3'
AND keyword.keyword IN ('SELECT keyword.keyword FROM keyword, movie_keyword WHERE keyword.id = movie_keyword.keyword_id GROUP BY keyword.keyword ORDER BY COUNT(*) DESC LIMIT 10000')
AND movie_info.info IN ('SELECT distinct movie_info.info from movie_info WHERE movie_info.info_type_id = 3')
AND kind_type.kind IN ('SELECT kind from kind_type')
