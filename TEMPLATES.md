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

* 2: Same join graph as 1, but added movie_keyword and keyword tables.
  - 2a: Generated using files in templates/toml2d
  - 2b: Generated using files in templates/toml2d2
  - 2c: Generated using files in templates/toml2dtitle

* 3:
  - 3a: Generated using files in templates/toml4

## Adding newer templates

TODO: add explanation
