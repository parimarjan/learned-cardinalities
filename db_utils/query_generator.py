from db_utils.utils import *

class QueryGenerator():
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
        self.query_template = query_template
        # key: column_name, val: [vals]
        self.valid_pred_vals = {}

        # tune-able params
        self.max_in_vals = 15

    def gen_queries(self, num_samples, column_stats=None):
        '''
        @ret: [sql queries]
        '''
        pred_columns, pred_types, pred_strs = extract_predicates(self.query_template)
        froms = extract_from_clause(self.query_template)
        joins = extract_join_clause(self.query_template)
        all_query_strs = []

        while len(all_query_strs) < num_samples:
            gen_query = self.query_template
            # now, replace each predicate value 1 by 1
            for i, col in enumerate(pred_columns):
                pred_str = pred_strs[i]
                if pred_types[i] == "eq":
                    pass
                elif pred_types[i] == "in":
                    # pdb.set_trace()
                    if not "SELECT" in pred_str[0]:
                        # leave this as is.
                        continue
                    pred_sql = pred_str[0]
                    if col not in self.valid_pred_vals:
                        output = cached_execute_query(pred_sql, self.user,
                                self.db_host, self.port, self.pwd, self.db_name,
                                100, None, None)
                        self.valid_pred_vals[col] = output


                    min_val = max(1, len(self.valid_pred_vals[col]) / 1000)
                    # replace pred_sql by a value from the valid ones
                    num_pred_vals = random.randint(min_val, self.max_in_vals)
                    # # find this many values randomly from the given col, and
                    # # update col_vals with it.
                    vals = ["'{}'".format(random.choice(self.valid_pred_vals[col])[0].replace("'",""))
                                for k in range(num_pred_vals)]
                    vals = [s for s in set(vals)]
                    vals.sort()
                    new_pred_str = ",".join(vals)
                    gen_query = gen_query.replace("'" + pred_sql + "'", new_pred_str)

                elif pred_types[i] == "lte" or pred_types[i] == "lt":
                    pass
                    # val1 = random.choice(col_all_vals[i])[0]
                    # val2 = random.choice(col_all_vals[i])[0]
                    # low_pred = "X" + col[col.find(".")+1:]
                    # high_pred = "Y" + col[col.find(".")+1:]
                    # low_val = str(min(val1, val2))
                    # high_val = str(max(val1, val2))
                    # cur_query = cur_query.replace(low_pred, low_val)
                    # cur_query = cur_query.replace(high_pred, high_val)
                    # cur_vals.append((low_val, high_val))

            all_query_strs.append(gen_query)
        print(len(all_query_strs))
        return all_query_strs
        # pdb.set_trace()

