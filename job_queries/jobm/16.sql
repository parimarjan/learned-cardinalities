
SELECT COUNT(*) FROM keyword AS k,kind_type AS kt,movie_info AS mi,movie_info_idx AS mi_idx,movie_keyword AS mk,title AS t WHERE t.production_year > 2010 AND k.keyword IN ('murder', 'murder-in-title') AND mi_idx.info > '6.0' AND kt.kind = 'movie' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND kt.id = t.kind_id AND t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mi_idx.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = mi_idx.movie_id AND mk.movie_id = mi_idx.movie_id AND mk.keyword_id = k.id