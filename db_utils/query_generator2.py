from db_utils.utils import *
import pdb

class QueryGenerator2():
    '''
    Generates sql queries based on a template.
    TODO: explain rules etc.
    '''
    def __init__(self, query_template, user, db_host, port,
            pwd, db_name):
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name
        # self.query_template = query_template
        self.base_sql = query_template["base_sql"]["sql"]
        self.templates = query_template["templates"]
        self.sampling_outputs = {}

        # tune-able params
        self.max_in_vals = 15

    def _update_sql_in(self, sql, samples, keys, pred_vals):

        for i, key in enumerate(keys):
            vals = []
            for s in samples:
                val = s[i]
                if val is not None:
                    vals.append("'{}'".format(val.replace("'","")))
                else:
                    assert False
            vals = [s for s in set(vals)]
            vals.sort()
            new_pred_str = ",".join(vals)
            sql = sql.replace(key, new_pred_str)
            pred_vals[key] = new_pred_str

        return sql

    def _gen_query_str(self, templated_preds):
        '''
        '''
        gen_sql = self.base_sql
        pred_vals = {}
        # for each group, select appropriate predicates
        for pred_group in templated_preds:
            print(pred_group["columns"])
            if "sql" in pred_group["type"]:
                if pred_group["type"] == "sqls":
                    cur_sql = random.choice(pred_group["sqls"])
                else:
                    cur_sql = pred_group["sql"]

                if pred_group["dependencies"]:
                    # need to replace placeholders in cur_sql
                    for fixed_key, fixed_pred in pred_vals.items():
                        cur_sql = cur_sql.replace(fixed_key, fixed_pred)
                cur_key = deterministic_hash(cur_sql)
                if cur_key in self.sampling_outputs:
                    output = self.sampling_outputs[cur_key]
                else:
                    output = cached_execute_query(cur_sql, self.user,
                            self.db_host, self.port, self.pwd, self.db_name,
                            100, None, None)
                    self.sampling_outputs[cur_key] = output

                if len(output) == 0:
                    return None

                # now use one of the different sampling methods
                num_samples = random.randint(pred_group["min_samples"],
                        pred_group["max_samples"])
                if pred_group["sampling_method"] == "quantile":
                    num_quantiles = pred_group["num_quantiles"]
                    curp = random.randint(0, num_quantiles-1)
                    chunk_len = int(len(output) / num_quantiles)
                    tmp_output = output[curp*chunk_len: (curp+1)*chunk_len]

                    samples = [random.choice(tmp_output) for _ in
                            range(num_samples)]
                    gen_sql = self._update_sql_in(gen_sql, samples,
                            pred_group["keys"], pred_vals)

                else:
                    samples = [random.choice(output) for _ in
                            range(num_samples)]
                    gen_sql = self._update_sql_in(gen_sql, samples,
                            pred_group["keys"], pred_vals)

                try:
                    print(samples)
                    total_count = sum([int(s[-1]) for s in samples])
                    print("total count: ", total_count)
                except:
                    pass

            elif pred_group["type"] == "list":
                if pred_group["sampling_method"] == "uniform":
                    if pred_group["pred_type"] == "range":
                        assert len(pred_group["keys"]) == 2
                        options = pred_group["options"]
                        pred_choice = random.choice(options)
                        for idx, val in enumerate(pred_choice):
                            key = pred_group["keys"][idx]
                            gen_sql = gen_sql.replace(key, str(val))
                            pred_vals[key] = str(val)
                    else:
                        assert len(pred_group["keys"]) == 1
                        options = pred_group["options"]
                        pred_choice = random.choice(options)
                        gen_sql = self._update_sql_in(gen_sql, [[pred_choice]],
                                pred_group["keys"], pred_vals)
            else:
                assert False

        return gen_sql

    def gen_queries(self, num_samples, column_stats=None):
        '''
        @ret: [sql queries]
        '''
        start = time.time()
        all_query_strs = []

        while len(all_query_strs) < num_samples:
            for template in self.templates:
                query_str = self._gen_query_str(template["predicates"])
                if query_str is not None:
                    print("generated query str: ")
                    print(query_str)
                    pdb.set_trace()
                    all_query_strs.append(query_str)

        print("{} took {} seconds to generate".format(len(all_query_strs),
            time.time()-start))
        return all_query_strs
