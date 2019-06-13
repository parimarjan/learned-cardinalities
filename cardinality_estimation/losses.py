import numpy as np
import pdb
import park
from utils.utils import *

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

def run_all_eps(env, fixed_agent=None):
    '''
    @ret: dict: query : info,
        where info is returned at the end of the episode by park. info should
        contain all the neccessary facts about that run.
    '''
    queries = {}
    while True:
        # don't know precise episode lengths, changes based on query, so use
        # the done signal to stop the episode
        done = False
        state = env.reset()
        query = env.get_current_query()
        # print(query)
        query = deterministic_hash(query)
        if query in queries.keys():
            # print("query already seen, breaking")
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

def compute_join_order_loss(alg, queries, use_subqueries,
        baseline="EXHAUSTIVE"):
    def update_cards(cardinalities, est_cards, q):
        for j, subq in enumerate(q.subqueries):
            # get all the tables in subquery
            tables = subq.table_names
            val = subq.true_count
            tables.sort()
            table_key = " ".join(tables)
            # ugh, initial space because of the way cardinalities json was
            # generated..
            table_key = " " + table_key
            cardinalities[i][table_key] = int(est_cards[j])

    assert len(queries[0].subqueries) > 0
    # create a new park env, and close at the end.
    env = park.make('query_optimizer')
    # Set queries
    query_dict = {}

    # TMP: debugging, hardcoded-queries
    # queries = queries[0:1]
    # fname = "/home/pari/query-optimizer/simple-queries/0.sql"
    # with open(fname, "r") as f:
        # queries[0].query = f.read()

    for i, q in enumerate(queries):
        query_dict[i] = q.query

    env.initialize_queries(query_dict)
    cardinalities = {}
    # Set estimated cardinalities
    for i, q in enumerate(queries):
        cardinalities[i] = {}
        yhat = alg.test(q.subqueries)
        yhat = np.array(yhat, dtype=np.float32)
        totals = np.array([q.total_count for q in q.subqueries],
                        dtype=np.float32)
        est_cards = np.multiply(yhat, totals)
        update_cards(cardinalities, est_cards, q)
        # print(cardinalities[i].keys())
    # pdb.set_trace()
    env.initialize_cardinalities(cardinalities)
    # print("num cards: ", len(cardinalities))
    # Learn optimal agent for estimated cardinalities
    agents = []
    train_q = run_all_eps(env)
    fixed_agent = {}
    for hashedq in train_q:
        info = train_q[hashedq]
        actions = info["joinOrders"][baseline]
        fixed_agent[hashedq] = actions

    assert len(fixed_agent) == len(cardinalities) == len(queries)
    agents.append(fixed_agent)

    cardinalities = {}
    # Set true cardinalities
    for i, q in enumerate(queries):
        cardinalities[i] = {}
        est_cards = np.array([q.true_count for q in q.subqueries])
        update_cards(cardinalities, est_cards, q)
    env.initialize_cardinalities(cardinalities)

    # Test agent on true cardinalities
    # TODO: optimize it so that exh search etc. don't have to rerun the
    # algorithm
    assert len(agents) == 1
    # TODO: save the data / compare across queries etc.
    # for rep, fixed_agent in enumerate(agents):
    fixed_agent = agents[0]

    test_q = run_all_eps(env, fixed_agent=fixed_agent)
    total_error = 0.00
    baseline_costs = []
    est_card_costs = []
    for q in test_q:
        info = test_q[q]
        bcost = info["costs"][baseline]
        card_cost = info["costs"]["RL"]
        cur_error = card_cost - bcost
        total_error += card_cost - bcost
        baseline_costs.append(float(bcost))
        est_card_costs.append(float(card_cost))

    total_avg_err = np.mean(np.array(est_card_costs)-np.array(baseline_costs))
    # print("total avg error: {}: {}".format(baseline, total_avg_err))
    rel_errors = np.array(est_card_costs) / np.array(baseline_costs)

    return rel_errors
