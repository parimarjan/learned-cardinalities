SELECT COUNT(*)
FROM
movie_keyword,
title,
keyword,
movie_info,
info_type,
kind_type,
movie_companies,
company_name
WHERE
title.id = movie_keyword.movie_id
AND movie_keyword.keyword_id = keyword.id
AND movie_info.movie_id = title.id
AND movie_info.movie_id = movie_keyword.movie_id
AND movie_info.info_type_id = info_type.id
AND title.kind_id = kind_type.id
AND title.id = movie_companies.movie_id
-- AND movie_companies.movie_id = movie_keyword.movie_id
-- AND movie_companies.movie_id = movie_info.movie_id
AND company_name.id = movie_companies.company_id
-- AND info_type.id = '3'
-- AND keyword.keyword IN ('SELECT keyword.keyword FROM keyword, movie_keyword WHERE keyword.id = movie_keyword.keyword_id AND keyword.keyword in (SELECT keyword.keyword FROM keyword, movie_keyword WHERE keyword.id = movie_keyword.keyword_id GROUP BY keyword.keyword ORDER BY count(*) DESC LIMIT 1000)')
-- AND movie_info.info IN ('SELECT movie_info.info from movie_info WHERE movie_info.info_type_id = 3')
-- AND kind_type.kind IN ('SELECT kind from kind_type')
-- AND 'Xproduction_year' < title.production_year
-- AND title.production_year <= 'Yproduction_year'
-- AND company_name.country_code IN ('SELECT country_code FROM company_name WHERE country_code is not NULL')
