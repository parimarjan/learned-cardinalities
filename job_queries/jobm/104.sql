
SELECT COUNT(*) FROM cast_info AS ci,title AS t WHERE t.production_year BETWEEN 1980 AND 1984 AND ci.movie_id = t.id