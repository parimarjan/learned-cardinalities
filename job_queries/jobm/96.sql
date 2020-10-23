
SELECT COUNT(*) FROM company_type AS ct,movie_companies AS mc,movie_info AS mi,title AS t WHERE t.production_year > 1990 AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND ct.kind = 'production companies' AND mc.note NOT LIKE '%(TV)%' AND mc.note LIKE '%(USA)%' AND t.id = mi.movie_id AND t.id = mc.movie_id AND mi.movie_id = mc.movie_id AND mc.company_type_id = ct.id