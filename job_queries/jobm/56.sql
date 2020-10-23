
SELECT COUNT(*) FROM cast_info AS ci,company_name AS cn,keyword AS k,movie_companies AS mc,movie_info AS mi,movie_keyword AS mk,title AS t WHERE t.production_year > 2010 AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND mi.info IS NOT NULL AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mc.company_id = cn.id AND ci.movie_id = mk.movie_id AND mk.keyword_id = k.id