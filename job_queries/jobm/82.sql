
SELECT COUNT(*) FROM cast_info AS ci,company_name AS cn,keyword AS k,movie_companies AS mc,movie_info AS mi,movie_info_idx AS mi_idx,movie_keyword AS mk,title AS t WHERE k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND cn.name LIKE 'Lionsgate%' AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mi_idx.movie_id = ci.movie_id AND mi_idx.movie_id = mk.movie_id AND mi_idx.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND ci.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id AND mk.keyword_id = k.id AND mc.company_id = cn.id