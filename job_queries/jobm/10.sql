
SELECT COUNT(*) FROM company_name AS cn,company_type AS ct,movie_companies AS mc,movie_info AS mi,movie_info_idx AS mi_idx,title AS t WHERE t.production_year BETWEEN 2000 AND 2010 AND mi_idx.info > '7.0' AND cn.country_code = '[us]' AND ct.kind = 'production companies' AND mi.info IN ('Drama', 'Horror', 'Western', 'Family') AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = mc.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id