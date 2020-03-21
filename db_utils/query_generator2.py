from db_utils.utils import *
import pdb
from nltk.tokenize import word_tokenize
import pygtrie

ILIKE_PRED_FMT = "'%{ILIKE_PRED}%'"
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
        self.ilike_output_size = {}
        self.bad_sqls = []

        self.max_in_vals = 15

    def _update_preds_range(self, sql, column, key, pred_val):
        '''
        '''
        print("key: ", key)
        print("pred_val: ", pred_val)
        pdb.set_trace()
        sql = sql.replace(key, pred_val)
        return sql

    def _generate_sql(self, pred_vals):
        '''
        '''
        sql = self.base_sql
        # for each group, select appropriate predicates
        for key, val in pred_vals.items():
            if key not in sql:
                print("key not in sql!")
                print(key)
                pdb.set_trace()
            sql = sql.replace(key, val)

        return sql

    def _update_sql_ilike(self, ilike_filter, pred_group, pred_vals):
        columns = pred_group["columns"]
        assert len(columns) == 1
        key = pred_group["keys"][0]
        pred_type = pred_group["pred_type"]

        pred_str = columns[0] + " " + pred_type + " " + ilike_filter
        pred_vals[key] = pred_str

    def _update_sql_in(self, samples, pred_group, pred_vals):
        '''
        @samples: ndarray, ith index correspond to possible values for ith
        index of pred_group["keys"], and last index is the count of the
        combined values.

        @pred_vals: all the predicates in the base sql string that have already
        been assigned a value. This will be updated after we assign values to
        the unspecified columns in the present pred_group.

        @pred_group: [[template.predicate]] section of the toml that this
        predicate corresponds to.

        @ret: updated sql string
        '''
        keys = pred_group["keys"]
        columns = pred_group["columns"]
        pred_type = pred_group["pred_type"]
        assert len(keys) == len(columns)

        for i, key in enumerate(keys):
            # key will be replaced with a predicate string
            pred_str = ""
            none_cond = None
            column = columns[i]
            vals = []
            # can have multiple values for IN statements, including None / NULL
            for s in samples:
                val = s[i]
                if val:
                    val = str(val)
                    vals.append("'{}'".format(val.replace("'","")))
                else:
                    # None value
                    none_cond = column + " IS NULL"

            vals = [s for s in set(vals)]
            if len(vals) == 0:
                assert none_cond
                pred_str = none_cond
            else:
                vals.sort()
                new_pred_str = ",".join(vals)
                pred_str = column + " " + pred_type + " "
                pred_str += "(" + new_pred_str + ")"
                if none_cond:
                    pred_str += " OR " + none_cond

            pred_vals[key] = pred_str

    def _gen_query_str(self, templated_preds):
        '''
        @templated_preds

        Modifies the base sql to plug in newer values at all the unspecified
        values.
            Handling of NULLs:

        '''
        # dictionary that is used to keep track of the column values that have
        # already been selected so far.
        pred_vals = {}

        # for each group, select appropriate predicates
        for pred_group in templated_preds:
            if "multi" in pred_group:
                # multiple predicate conditions, choose any one
                pred_group = random.choice(pred_group["multi"])

            if "sql" in pred_group["type"]:
                # cur_sql will be the sql used to sample for this predicate
                # value
                if pred_group["type"] == "sqls":
                    cur_sql = random.choice(pred_group["sqls"])
                else:
                    cur_sql = pred_group["sql"]

                if pred_group["dependencies"]:
                    # need to replace placeholders in cur_sql
                    for key, val in pred_vals.items():
                        cur_sql = cur_sql.replace(key, val)

                # get possible values to use
                cur_key = deterministic_hash(cur_sql)
                if cur_key in self.sampling_outputs:
                    output = self.sampling_outputs[cur_key]
                else:
                    if cur_sql in self.bad_sqls:
                        return None

                    output, _ = cached_execute_query(cur_sql, self.user,
                            self.db_host, self.port, self.pwd, self.db_name,
                            100, "./qgen_cache", None)
                    if pred_group["pred_type"].lower() == "ilike":
                        print("going to tokenize all words")
                        tokens = []
                        for out in output:
                            cur_tokens = word_tokenize(out[0].lower())
                            if len(cur_tokens) > 15:
                                print("too many tokens in column: ",
                                        pred_group["columns"][0], pred_vals)
                                self.bad_sqls.append(cur_sql)
                                return None
                            tokens += cur_tokens

                        print("going to make a trie..")
                        trie = pygtrie.CharTrie()
                        for token in tokens:
                            if token in trie:
                                trie[token] += 1
                            else:
                                trie[token] = 1

                        self.ilike_output_size[cur_key] = len(output)
                        self.sampling_outputs[cur_key] = trie
                        output = trie
                    else:
                        self.sampling_outputs[cur_key] = output

                if len(output) == 0:
                    # no point in doing shit
                    return None

                if pred_group["pred_type"].lower() == "in":
                    # now use one of the different sampling methods
                    num_samples = random.randint(pred_group["min_samples"],
                            pred_group["max_samples"])

                    if pred_group["sampling_method"] == "quantile":
                        num_quantiles = pred_group["num_quantiles"]
                        curp = random.randint(0, num_quantiles-1)
                        chunk_len = int(len(output) / num_quantiles)
                        tmp_output = output[curp*chunk_len: (curp+1)*chunk_len]
                        if len(tmp_output) == 0:
                            # really shouldn't be happenning right?
                            return None

                        if len(tmp_output) <= num_samples:
                            samples = [random.choice(tmp_output) for _ in
                                    range(num_samples)]
                        else:
                            samples = random.sample(tmp_output, num_samples)

                        self._update_sql_in(samples,
                                pred_group, pred_vals)

                    else:
                        samples = [random.choice(output) for _ in
                                range(num_samples)]
                        self._update_sql_in(samples,
                                pred_group, pred_vals)

                elif pred_group["pred_type"].lower() == "ilike":
                    assert isinstance(output, pygtrie.CharTrie)

                    # Note: the trie will only provide a lower-bound on the
                    # number of matches, since ILIKE predicates would also
                    # consider substrings. But this seems to be enough for our
                    # purposes, as we will avoid queries that zero out
                    num_rows = self.ilike_output_size[cur_key]
                    # choose a random key
                    partition_size = num_rows / pred_group["num_quantiles"]
                    cur_partition = random.randint(0, pred_group["num_quantiles"]-1)
                    min_target = max(pred_group["min_count"],
                            cur_partition*partition_size)
                    max_target = (cur_partition+1)*partition_size

                    # 10 tries to find appropriate filter
                    for i in range(20):
                        key = random.choice(output.keys())
                        if len(key) < pred_group["min_chars"]:
                            continue
                        max_filter_len = min(len(key), pred_group["max_chars"])
                        num_chars = random.randint(pred_group["min_chars"],
                                max_filter_len)
                        ilike_pred = key[0:num_chars]
                        est_size = sum(output[ilike_pred:])
                        if est_size > min_target and est_size < max_target:
                            break
                        else:
                            ilike_pred = None

                    if ilike_pred is None:
                        print("did not find an appropriate predicate for ",
                                pred_group["columns"])
                        return None
                    else:
                        print("col: {}, quantile: {}, filter: {}, est size: {}".format(
                            pred_group["columns"][0], cur_partition, ilike_pred, est_size))
                    ilike_filter = ILIKE_PRED_FMT.format(ILIKE_PRED = ilike_pred)
                    self._update_sql_ilike(ilike_filter, pred_group, pred_vals)
                else:
                    assert False

            elif pred_group["type"] == "list":
                ## assuming it is a single column
                columns = pred_group["columns"]
                assert len(columns) == 1
                if pred_group["sampling_method"] == "uniform":
                    if pred_group["pred_type"] == "range":
                        col = columns[0]
                        assert len(pred_group["keys"]) == 2
                        options = pred_group["options"]
                        pred_choice = random.choice(options)
                        assert len(pred_choice) == 2
                        lower_key = pred_group["keys"][0]
                        upper_key = pred_group["keys"][1]
                        lower_val = pred_choice[0]
                        upper_val = pred_choice[1]

                        assert len(pred_choice) == 2
                        if "numeric_col_type" in pred_group:
                            col_type = pred_group["numeric_col_type"]
                            # add a chec for both conditions
                            float_regex = '^(?:[1-9]\d*|0)?(?:\.\d+)?$'
                            num_check_cond_tmp = "{col} ~ '{regex}' AND {cond}"

                            upper_cond = "{val} <= {col}::{col_type}".format(col=col,
                                                                val=lower_val,
                                                                col_type=col_type)
                            lower_cond = "{col}::{col_type} <= {val}".format(col=col,
                                                                val=upper_val,
                                                                col_type=col_type)
                            lower_cond = num_check_cond_tmp.format(col=col,
                                                        cond = lower_cond,
                                                        regex = float_regex)
                            upper_cond = num_check_cond_tmp.format(col=col,
                                                    cond = upper_cond, regex =
                                                    float_regex)
                        else:
                            lower_key = pred_group["keys"][0]
                            upper_key = pred_group["keys"][1]
                            lower_val = pred_choice[0]
                            upper_val = pred_choice[1]
                            lower_cond = "{} >= {}".format(col, lower_val)
                            upper_cond = "{} <= {}".format(col, upper_val)

                        pred_vals[lower_key] = lower_cond
                        pred_vals[upper_key] = upper_cond

                    else:
                        options = pred_group["options"]
                        pred_choice = random.choice(options)
                        if "replace" in pred_group:
                            # assert len(pred_choice) == 1
                            assert len(pred_group["keys"]) == 1
                            # cur_choice = pred_choice[0]
                            cur_key = pred_group["keys"][0]
                            pred_vals[cur_key] = pred_choice
                            pdb.set_trace()
                        else:
                            # probably only deals with `=` ?
                            assert len(pred_group["keys"]) == 1
                            self._update_sql_in([[pred_choice]],
                                    pred_group, pred_vals)
            else:
                assert False

        gen_sql = self._generate_sql(pred_vals)
        return gen_sql

    def gen_queries(self, num_samples, column_stats=None):
        '''
        @ret: [sql queries]
        '''
        print("going to generate ", num_samples)
        start = time.time()
        all_query_strs = []

        while len(all_query_strs) < num_samples:
            for template in self.templates:
                query_str = self._gen_query_str(template["predicates"])
                if query_str is not None:
                    all_query_strs.append(query_str)
                    print(query_str)
                    # pdb.set_trace()
                else:
                    print("query str was None")

        print("{} took {} seconds to generate".format(len(all_query_strs),
            time.time()-start))
        return all_query_strs
