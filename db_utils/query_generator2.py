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
        self.query_template = query_template
        # key: column_name, val: [vals]
        self.valid_pred_vals = {}

        # tune-able params
        self.max_in_vals = 15

    def gen_queries(self, num_samples, column_stats=None):
        '''
        @ret: [sql queries]
        '''
        start = time.time()
        all_query_strs = []

        print(self.query_template)
        pdb.set_trace()

        while len(all_query_strs) < num_samples:
            pass

        print("{} took {} seconds to generate".format(len(all_query_strs),
            time.time()-start))
        return all_query_strs

