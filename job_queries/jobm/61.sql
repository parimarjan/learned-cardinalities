
SELECT COUNT(*) FROM cast_info AS ci,comp_cast_type AS cct1,complete_cast AS cc,keyword AS k,kind_type AS kt,movie_info_idx AS mi_idx,movie_keyword AS mk,title AS t WHERE t.production_year > 2000 AND cct1.kind = 'cast' AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence', 'magnet', 'web', 'claw', 'laser') AND mi_idx.info > '7.0' AND kt.kind = 'movie' AND kt.id = t.kind_id AND t.id = mk.movie_id AND t.id = ci.movie_id AND t.id = cc.movie_id AND t.id = mi_idx.movie_id AND mk.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND mk.movie_id = mi_idx.movie_id AND mk.keyword_id = k.id AND ci.movie_id = cc.movie_id AND ci.movie_id = mi_idx.movie_id AND cc.movie_id = mi_idx.movie_id AND cc.subject_id = cct1.id