SELECT COUNT(*)
FROM
title,
kind_type,
movie_link,
link_type,
movie_companies,
company_type,
comp_cast_type,
complete_cast
WHERE title.id = movie_link.movie_id
AND title.id = movie_companies.movie_id
AND title.kind_id = kind_type.id
AND title.id = complete_cast.movie_id
AND complete_cast.subject_id = comp_cast_type.id
AND movie_companies.company_type_id = company_type.id
AND movie_link.link_type_id = link_type.id
AND movie_companies.movie_id = movie_link.movie_id
AND complete_cast.movie_id = movie_link.movie_id
AND complete_cast.movie_id = movie_companies.movie_id
AND link_type.link IN ('SELECT link from link_type')
AND comp_cast_type.kind IN ('SELECT kind from comp_cast_type')
AND company_type.kind IN ('SELECT kind from company_type')
AND kind_type.kind IN ('SELECT kind from kind_type')
AND 'Xproduction_year' < title.production_year
AND title.production_year <= 'Yproduction_year'
