
SELECT COUNT(*) FROM cast_info AS ci,keyword AS k,movie_keyword AS mk,title AS t WHERE k.keyword = 'marvel-cinematic-universe' AND t.production_year > 2014 AND k.id = mk.keyword_id AND mk.movie_id = t.id AND mk.movie_id = ci.movie_id AND t.id = ci.movie_id