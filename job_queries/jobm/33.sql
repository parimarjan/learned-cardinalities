
SELECT COUNT(*) FROM cast_info AS ci,movie_info AS mi,movie_info_idx AS mi_idx,title AS t WHERE t.production_year BETWEEN 2008 AND 2014 AND mi_idx.info > '8.0' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mi.info IN ('Horror', 'Thriller') AND mi.note IS NULL AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id