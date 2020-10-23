
SELECT COUNT(*) FROM cast_info AS ci,company_name AS cn,movie_companies AS mc,title AS t WHERE cn.country_code = '[us]' AND t.production_year BETWEEN 2007 AND 2010 AND ci.note = '(voice)' AND mc.note LIKE '%(200%)%' AND ci.movie_id = t.id AND ci.movie_id = mc.movie_id AND t.id = mc.movie_id AND mc.company_id = cn.id