SELECT COUNT(*)
FROM
movie_keyword,
title,
keyword,
movie_info,
info_type,
kind_type,
cast_info,
name,
role_type
WHERE
movie_keyword.movie_id = title.id
AND movie_info.movie_id = title.id
AND cast_info.movie_id = title.id
AND movie_keyword.movie_id = movie_info.movie_id
AND cast_info.movie_id = movie_info.movie_id
AND role_type.id = cast_info.role_id
AND cast_info.person_id = name.id
AND movie_keyword.keyword_id = keyword.id
AND movie_info.info_type_id = info_type.id
AND title.kind_id = kind_type.id
AND info_type.id = '3'
and keyword.keyword in ('select keyword.keyword from keyword, movie_keyword where keyword.id = movie_keyword.keyword_id and keyword.keyword in (select keyword.keyword from keyword, movie_keyword where keyword.id = movie_keyword.keyword_id group by keyword.keyword order by count(*) desc limit 1000)')
AND movie_info.info IN ('SELECT movie_info.info from movie_info WHERE movie_info.info_type_id = 3')
AND kind_type.kind IN ('SELECT kind from kind_type')
AND role_type.role IN ('SELECT role from role_type')
AND name.gender IN ('SELECT distinct gender from name WHERE gender is not NULL')
AND 'Xproduction_year' < title.production_year
AND title.production_year <= 'Yproduction_year'
