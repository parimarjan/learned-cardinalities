
SELECT COUNT(*) FROM aka_title AS AT,company_name AS cn,company_type AS ct,keyword AS k,movie_companies AS mc,movie_info AS mi,movie_keyword AS mk,title AS t WHERE t.production_year > 1990 AND cn.country_code = '[us]' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND t.id = at.movie_id AND t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND at.movie_id = mk.movie_id AND at.movie_id = mi.movie_id AND at.movie_id = mc.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id AND mk.keyword_id = k.id AND mc.company_id = cn.id AND mc.company_type_id = ct.id