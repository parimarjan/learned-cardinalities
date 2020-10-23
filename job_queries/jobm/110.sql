
SELECT COUNT(*) FROM cast_info AS ci,company_name AS cn,movie_companies AS mc,title AS t WHERE cn.country_code = '[us]' AND t.production_year BETWEEN 2005 AND 2015 AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND ci.movie_id = t.id AND ci.movie_id = mc.movie_id AND t.id = mc.movie_id AND mc.company_id = cn.id