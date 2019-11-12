
Postgres CM instructions:
  - use my fork of pg_hint_plan
    - switch branch, make, make install
  - postgresql.conf:
    - switch off parallelism (or not..)
    - geqo = 0
  - relevant new flags:
    - --jl_use_postgres 1 --qopt_use_java 0

Join Loss Viz:
