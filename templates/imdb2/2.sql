SELECT COUNT(*)
FROM
title,
kind_type,
cast_info,
role_type,
movie_info,
info_type,
name
WHERE title.id = cast_info.movie_id
AND cast_info.role_id = role_type.id
AND title.kind_id = kind_type.id
AND name.id = cast_info.person_id
AND movie_info.movie_id = title.id
AND movie_info.info_type_id = info_type.id
AND info_type.id = '3'
AND 'Xproduction_year' < title.production_year
AND title.production_year <= 'Yproduction_year'
AND role_type.role IN ('SELECT role from role_type')
AND kind_type.kind IN ('SELECT kind from kind_type')
AND name.gender IN ('SELECT DISTINCT gender FROM name WHERE gender is not null')
AND movie_info.info IN ('SELECT movie_info.info from movie_info WHERE movie_info.info_type_id = 3')
