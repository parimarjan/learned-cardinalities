
SELECT COUNT(*) FROM cast_info AS ci,company_name AS cn,movie_companies AS mc,movie_info AS mi,title AS t WHERE t.production_year BETWEEN 2005 AND 2009 AND mc.note IS NOT NULL AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND mi.info IS NOT NULL AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id