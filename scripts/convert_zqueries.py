import json
import pdb
import os

table_aliases = {}
table_aliases["title"] = "t"
table_aliases["cast_info"] = "ci"
table_aliases["movie_info"] = "mi"
table_aliases["movie_info_idx"] = "mii"
table_aliases["person_info"] = "pi"
table_aliases["name"] = "n"
table_aliases["aka_name"] = "an"
table_aliases["keyword"] = "k"
table_aliases["movie_keyword"] = "mk"
table_aliases["movie_companies"] = "mc"
table_aliases["movie_link"] = "ml"
table_aliases["aka_title"] = "at"
table_aliases["complete_cast"] = "cc"
table_aliases["kind_type"] = "kt"
table_aliases["role_type"] = "rt"
table_aliases["char_name"] = "chn"
table_aliases["info_type"] = "it"
table_aliases["company_type"] = "ct"
table_aliases["company_name"] = "cn"
table_aliases["movie_link"] = "ml"
table_aliases["link_type"] = "lt"
table_aliases["comp_cast_type"] = "cct"

FROM_FMT = "{} AS {}"
SQL_FMT = "{AGG} FROM {FROMS} WHERE {WHERES}"

# FN = "/flash1/ziniuw/zero-shot-data/runs/raw/imdb_full/complex_queries_training_50k.json"
FN = "/flash1/ziniuw/zero-shot-data/runs/raw/imdb_full/complex_queries_testing_2k.json"
OUTDIR = "./queries/zero-shot-test/all/"

with open(FN, "r") as f:
    data = json.load(f)

print(len(data))

for di, d in enumerate(data):
    sql = d[0]
    ons = sql.count("ON")
    # if ons == 0:
        # continue

    # if "FROM" not in sql or "WHERE" not in sql or "AND" not in sql:
    # if "FROM" not in sql or "WHERE" not in sql:
        # print(sql)
        # continue

    if "FROM" not in sql:
        print(sql)
        continue

    agg_sql = sql[0:sql.find("FROM")]
    from_sql = sql[sql.find("FROM")+4: sql.find("WHERE")]

    if "LEFT OUTER JOIN" in from_sql:
        joins = from_sql.split("LEFT OUTER JOIN")
    elif "INNER JOIN" in from_sql:
        print(from_sql)
        pdb.set_trace()
        joins = from_sql.split("INNER JOIN")
    elif " JOIN " in from_sql:
        joins = from_sql.split("JOIN")
    else:
        pass

    tables = set()
    where_conds = []

    for j in joins:
        curjoins = j.split("ON")
        if len(curjoins) > 2:
            print(curjoins)
            pdb.set_trace()
        table = curjoins[0]
        table = table.replace(" ", "")
        tables.add(table)

        if len(curjoins) <= 1:
            continue

        curjoin = curjoins[1]
        curjoin = curjoin.replace(" ", "")

        for tab in tables:
            alias = table_aliases[tab.replace('"', '')]
            curjoin = curjoin.replace(tab, alias)

        where_conds.append(curjoin)

    for tab in tables:
        alias = table_aliases[tab.replace('"', '')]
        agg_sql = agg_sql.replace(tab, alias)

    from_conds = []
    for tab in tables:
        # tab = tab.replace('"', '')
        alias = table_aliases[tab.replace('"', '')]
        from_conds.append(FROM_FMT.format(tab, alias))

    where_sql = sql[sql.find("WHERE")+5:]
    where_sql = where_sql.split("AND")

    for wcond in where_sql:
        for tab in tables:
            alias = table_aliases[tab.replace('"', '')]
            wcond = wcond.replace(tab+".", alias+".")

        where_conds.append(wcond)

    from_sql = ",".join(from_conds)
    where_sql = " AND ".join(where_conds)
    new_sql = SQL_FMT.format(AGG = agg_sql, FROMS = from_sql,
                   WHERES = where_sql)
    new_sql = new_sql.replace("'\'", "")
    new_sql = new_sql.replace("\n", "")

    outfn = os.path.join(OUTDIR, str(di) + ".sql")

    with open(outfn, "w") as f:
        f.write(new_sql)

    new_sql = new_sql.replace(";", "")

pdb.set_trace()
