SELECT COUNT(*)
FROM
title,
kind_type,
cast_info,
role_type,
name,
keyword,
movie_keyword
WHERE title.id = cast_info.movie_id
AND cast_info.role_id = role_type.id
AND title.kind_id = kind_type.id
AND name.id = cast_info.person_id
AND movie_keyword.movie_id = title.id
AND keyword.id = movie_keyword.keyword_id
AND 'Xproduction_year' < title.production_year
AND title.production_year <= 'Yproduction_year'
AND role_type.role IN (Xrole)
AND kind_type.kind IN (Xkind)
AND keyword.keyword IN (Xkeyword)
AND name.gender IN (Xgender)
