import numpy as np
import pdb
import park
from utils.utils import *
from cardinality_estimation.query import *

EPSILON = 0.000001
REL_LOSS_EPSILON = EPSILON
QERR_MIN_EPS = EPSILON

def get_loss(loss):
    if loss == "abs":
        return compute_abs_loss
    elif loss == "rel":
        return compute_relative_loss
    elif loss == "qerr":
        return compute_qerror
    elif loss == "join-loss":
        return compute_join_order_loss
    else:
        assert False

def get_loss_name(loss_name):
    if "qerr" in loss_name:
        return "qerr"
    elif "join" in loss_name:
        return "join"
    elif "abs" in loss_name:
        return "abs"
    elif "rel" in loss_name:
        return "rel"

def get_all_subqueries(queries):
    # FIXME: don't remember if the full query, Q, is part of subqueries or not.
    new_queries = []
    for q in queries:
        new_queries.append(q)
        for sq in q.subqueries:
            new_queries.append(sq)
    return new_queries

# TODO: put the yhat, ytrue parts in db_utils
def compute_relative_loss(alg, queries, use_subqueries, **kwargs):
    '''
    as in the quicksel paper.
    '''
    if use_subqueries:
        queries = get_all_subqueries(queries)
    yhat = alg.test(queries)
    yhat = np.array(yhat)
    ytrue = np.array([s.true_sel for s in queries])
    epsilons = np.array([REL_LOSS_EPSILON]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    errors = np.abs(ytrue - yhat) / ytrue
    return errors

def compute_abs_loss(alg, queries, use_subqueries, **kwargs):
    if use_subqueries:
        queries = get_all_subqueries(queries)
    yhat = alg.test(queries)
    ytrue = np.array([t.true_count for t in queries],
            dtype=np.float32)
    yhat = np.array(yhat, dtype=np.float32)
    totals = np.array([q.total_count for q in queries],
                    dtype=np.float32)
    yhat_total = np.multiply(yhat, totals)
    errors = np.abs(yhat_total - ytrue)
    return errors

def compute_qerror(alg, queries, use_subqueries, **kwargs):
    if use_subqueries:
        queries = get_all_subqueries(queries)
    yhat = alg.test(queries)
    ytrue = [s.true_sel for s in queries]
    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)
    errors = np.maximum( (ytrue / yhat), (yhat / ytrue))
    return errors

def run_all_eps(env, num_queries, fixed_agent=None):
    '''
    @ret: dict: query : info,
        where info is returned at the end of the episode by park. info should
        contain all the neccessary facts about that run.
    '''
    queries = {}
    # print("run all eps, num = ", num_queries)
    while True:
        # print("len queries: ", len(queries))
        if len(queries) >= num_queries:
            # HACK: avoid running the random episode as below
            break
        # don't know precise episode lengths, changes based on query, so use
        # the done signal to stop the episode
        done = False
        state = env.reset()
        # query = env.get_current_query()
        # query = deterministic_hash(query)
        query = env.get_current_query_name()
        if query in queries.keys():
            # FIXME: ugly hack, so we don't leave an episode hanging midway
            # env._run_random_episode()
            print("SHOULD NEVER HAVE HAPPPNED!")
            pdb.set_trace()
            assert False
            break
        # episode loop
        num_ep = 0
        while not done:
            if fixed_agent is None:
                action = env.action_space.sample()
            else:
                action = tuple(fixed_agent[query][num_ep])
            new_state, reward, done, info = env.step(action)
            state = new_state
            num_ep += 1
        queries[query] = info
    return queries

# FIXME: make this its own function.
def update_cards(est_cards, q, fix_aliases=True):
    '''
    @est_cards: cardinalities for each of the subqueries in q.subqueries.
    @ret: cardinalities for each subquery.

    HACK: because alias information is being lost in calcite, we hack table
    name + filter value as key (ugh....)
    '''
    cards = {}
    for j, subq in enumerate(q.subqueries):
        # get all the tables in subquery
        tables = subq.table_names
        filter_tables = []
        try:
            assert hasattr(subq, "froms")
        except:
            print(subq)
            pdb.set_trace()

        if fix_aliases:
            # STUPID hack.
            aliases = [a for a in subq.aliases]
            for k, alias in enumerate(aliases):
                # deal with multiple predicates ...
                seen = set()
                table = subq.aliases[alias]
                # because range queries have additional entry
                # update table[k] to also include filter, if a filter is present.
                # alias = get_alias(subq, table)

                # check each of the predicates to see if alias present
                val = ""
                for pred_i, pred in enumerate(subq.pred_column_names):
                    if pred in seen:
                        continue
                    # yikes
                    if alias == "it1":
                        val = str(3)
                        break
                    elif alias == "it2":
                        val = str(4)
                        break
                    # kill me
                    if " " + alias + "." in " " + pred:
                        # ...assuming it is sorted here...
                        if subq.cmp_ops[pred_i] == "lt":
                            val = str(subq.vals[pred_i][1])
                        else:
                            val = subq.vals[pred_i][0]

                    seen.add(pred)

                filter_tables.append(table + val)

        val = subq.true_count
        tables.sort()
        filter_tables.sort()
        table_key = " ".join(tables)
        filter_table_key = " ".join(filter_tables)
        # ugh, initial space because of the way cardinalities json was
        # generated..
        # table_key = " " + table_key

        ## FIXME: shouldn't need both these, but need to check into calcite..
        cards[table_key] = int(est_cards[j])
        cards[filter_table_key] = int(est_cards[j])

    return cards

def join_loss(pred, queries, old_env,
        baseline="EXHAUSTIVE"):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    '''
    if old_env is None:
        env = park.make('query_optimizer')
    else:
        env = old_env

    start = time.time()
    assert len(queries[0].subqueries) > 0
    # Set queries
    query_dict = {}

    # each queries index is set to its name
    for i, q in enumerate(queries):
        query_dict[str(deterministic_hash(q.query))] = q.query

    # env.initialize_queries(query_dict)
    print("initialized queries")
    cardinalities = {}
    # Set estimated cardinalities. For estimated cardinalities, we need to
    # add ONLY the subquery cardinalities
    pred_start = 0
    for i, q in enumerate(queries):
        # skip the first query, since that is not a subquery
        pred_start += 1
        yhat = []
        # this loop depends on the fact that pred[0],
        # pred[0+len(q[0].subqueries)]], etc would be the cardinalities for the
        # full query objects
        for j in range(pred_start, pred_start+len(q.subqueries), 1):
            yhat.append(pred[j])
        yhat = np.array(yhat, dtype=np.float32)
        assert len(yhat) == len(q.subqueries)
        totals = np.array([q.total_count for q in q.subqueries],
                        dtype=np.float32)
        est_cards = np.multiply(yhat, totals)
        # cardinalities[str(i)] = update_cards(est_cards, q)
        cardinalities[str(deterministic_hash(q.query))] = update_cards(est_cards, q)
        pred_start += len(q.subqueries)

    # Set true cardinalities
    true_cardinalities = {}
    for i, q in enumerate(queries):
        est_cards = np.array([q.true_count for q in q.subqueries])
        # true_cardinalities[str(i)] = update_cards(est_cards, q)
        true_cardinalities[str(deterministic_hash(q.query))] = update_cards(est_cards, q)

    print("both true cardinalities, and estimated cardinalities calculated")
    est_card_costs_dict, baseline_costs_dict = \
                env.compute_join_order_loss(query_dict, true_cardinalities, cardinalities,
                        baseline)
    est_card_costs = []
    baseline_costs = []

    # need to convert it back into arrays
    for i, q in enumerate(queries):
        key = str(deterministic_hash(q.query))
        est_card_costs.append(est_card_costs_dict[key])
        baseline_costs.append(baseline_costs_dict[key])

    if old_env is None:
        env.clean()

    return est_card_costs, baseline_costs

def compute_join_order_loss(alg, queries, use_subqueries,
        baseline="EXHAUSTIVE", compute_runtime=False):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    '''
    start = time.time()
    all_queries = []
    for i, q in enumerate(queries):
        all_queries.append(q)
        all_queries += q.subqueries
    pred = alg.test(all_queries)

    env = park.make('query_optimizer')
    est_card_costs, baseline_costs = join_loss(pred, queries, env, baseline=baseline)

    env.clean()
    return np.array(est_card_costs) - np.array(baseline_costs)
