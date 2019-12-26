# Query Templates

All queries are generated using templates defined in /templates.

## Query Format:

TODO: explain the format used in sql_representation.

## Query Name Format:

Each unique query generated using our templates are identified using the
following format:
  template_number + template_variant + query_num e.g., 1a1, 2b10 etc.

* template number: 1...n. Uniquely identifies the join graph in the queries
* template variant: a...z. For a given join graph, the predicates may still be
on different columns. Different template files are used to generate such
variants. e.g., queries 2a1,2a100, 2b1, 2b100 etc.
* query_num: 1...n

## Template Details

* 1:
  1a: Generated using files in: templates/toml2b/

  ```sql
  SELECT COUNT(*) FROM title as t,
  kind_type as kt,
  movie_info as mi1,
  info_type as it1,
  movie_info as mi2,
  info_type as it2,
  cast_info as ci,
  role_type as rt,
  name as n
  WHERE
  t.id = ci.movie_id
  AND t.id = mi1.movie_id
  AND t.id = mi2.movie_id
  AND mi1.movie_id = mi2.movie_id
  AND mi1.info_type_id = it1.id
  AND mi2.info_type_id = it2.id
  AND it1.id = '3'
  AND it2.id = '4'
  AND t.kind_id = kt.id
  AND ci.person_id = n.id
  AND ci.role_id = rt.id
  AND mi1.info IN (Xgenre)
  AND mi2.info IN (Xlanguage)
  AND kt.kind IN (Xmovie_kind)
  AND rt.role IN (Xrole)
  AND n.gender IN (Xgender)
  AND t.production_year <= Xprod_year_up
  AND Xprod_year_low < t.production_year
  ```

* 2: Same join graph as 1, but added movie_keyword and keyword tables.
  - 2a: Generated using files in templates/toml2d
  - 2b: Generated using files in templates/toml2d2
  - 2c: Generated using files in templates/toml2dtitle

* 3:
  - 3a: Generated using files in templates/toml4

## Adding newer templates

TODO: add explanation
