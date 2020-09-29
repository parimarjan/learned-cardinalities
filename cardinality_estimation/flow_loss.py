from torch.autograd import Function
from utils.utils import *
from db_utils.utils import *
import torch
import time
import pdb
from multiprocessing import Pool
from scipy import sparse

import platform
from ctypes import *
import os
import copy
import pkg_resources
import jax

# import jax.numpy as jp
# from jax import grad, jit, vmap
import jax.numpy as jp
from jax import jacfwd, jacrev

system = platform.system()
lib_dir = "./flow_loss_cpp"
if system == 'Linux':
    lib_file = "libflowloss.so"
    lib_file = lib_dir + "/" + lib_file
    fl_cpp = CDLL(lib_file, mode=RTLD_GLOBAL)
else:
    print("flow loss C library not being used as we are not on linux")
    # lib_file = "libflowloss.dylib"

DEBUG_JAX = False
DEBUG = False

def get_costs_jax(card1, card2, card3, nilj, cost_model,
        total1, total2,penalty):
    assert cost_model == "nested_loop_index7"
    if cost_model == "cm1":
        # hash_join_cost = card1 + card2
        if nilj == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False
            # nilj_cost = 10000000000
        # cost = jp.min(hash_join_cost, nilj_cost)
        # if cost != nilj_cost:
            # print("hash join cost selected!")
            # pdb.set_trace()
        cost = nilj_cost
    elif cost_model == "cm2":
        cost = CARD_DIVIDER*card1 + CARD_DIVIDER*card2
    elif cost_model == "nested_loop_index":
        if nilj == 1:
            # using index on node1
            ratio_mul = max(card3 / card2, 1.0)
            cost = NILJ_CONSTANT2*card2*ratio_mul
        elif nilj == 2:
            # using index on node2
            ratio_mul = max(card3 / card1, 1.0)
            cost = NILJ_CONSTANT2*card1*ratio_mul
        else:
            assert False
            # cost = card1*card2
        assert cost >= 1.0
    elif cost_model == "nested_loop_index2":
        if nilj == 1:
            # using index on node1
            cost = NILJ_CONSTANT2*card2
        elif nilj == 2:
            # using index on node2
            cost = NILJ_CONSTANT2*card1
        else:
            assert False
        assert cost >= 1.0
    elif cost_model == "nested_loop_index3":
        if nilj == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False
        cost = nilj_cost
    elif cost_model == "nested_loop_index7":

        if nilj == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1;
        elif nilj == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2;
        elif nilj == 3:
            # nilj_cost = card1*card2 + 100000;
            nilj_cost = card1 + card2;
        else:
            assert False
        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop_index7b":
        if nilj == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1;
        elif nilj == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2;
        else:
            assert False

        cost = nilj_cost

    elif cost_model == "nested_loop_index4":
        # same as nested_loop_index, but also considering just joining the two
        # tables

        # because card3 for ratio_mul should be calculated without applying the
        # predicate on the table with the index, and we don't have that value,
        # we replace it with a const
        if nilj == 1:
            # using index on node1
            ratio_mul = 1.0
            if (card3 / card2) > 1.0:
                ratio_mul = card3 / card2
            ratio_mul = ratio_mul*RATIO_MUL_CONST
            cost = NILJ_CONSTANT2*card2*ratio_mul
        elif nilj == 2:
            # using index on node2
            ratio_mul = 1.0
            if (card3 / card1) > 1.0:
                ratio_mul = card3 / card1
            ratio_mul = ratio_mul*RATIO_MUL_CONST
            cost = NILJ_CONSTANT2*card1*ratio_mul
        else:
            assert False
            # cost = card1*card2
        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost:
            print("cost2 < cost!")
            cost = cost2

        assert cost >= 1.0
    elif cost_model == "nested_loop_index5":
        # same as nested_index_loop4, BUT disallowing index joins if either of
        # the nodes have cardinality less than NILJ_MIN_CARD
        if card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD:
            cost = 1e10
        else:
            if nilj == 1:
                # using index on node1
                ratio_mul = 1.0
                if (card3 / card2) > 1.0:
                    ratio_mul = card3 / card2
                ratio_mul = ratio_mul*RATIO_MUL_CONST
                cost = NILJ_CONSTANT2*card2*ratio_mul
            elif nilj == 2:
                # using index on node2
                ratio_mul = 1.0
                if (card3 / card1) > 1.0:
                    ratio_mul = card3 / card1
                ratio_mul = ratio_mul*RATIO_MUL_CONST
                cost = NILJ_CONSTANT2*card1*ratio_mul
            else:
                assert False

        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost or (card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD):
            print("cost2 chosen! ", cost2)
            cost = cost2
    elif cost_model == "nested_loop_index6":
        if card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD:
            cost = 1e10
        else:
            if card1 > card2:
                cost = NILJ_CONSTANT2*card1*RATIO_MUL_CONST
            else:
                cost = NILJ_CONSTANT2*card2*RATIO_MUL_CONST

        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost or (card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD):
            print("cost2 chosen! ", cost2)
            cost = cost2
    elif cost_model == "nested_loop_index8":
        # same as nli7 --> but consider the fact the right side of an index
        # nested loop join WILL not have predicates pushed down
        # also, remove the term for index entirely
        if nilj == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est

        elif nilj == 2:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        # TODO: we may be doing fine without this one
        cost2 = SEQ_CONST*card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop_index8b":
        # same as nli7 --> but consider the fact the right side of an index
        # nested loop join WILL not have predicates pushed down
        # also, remove the term for index entirely
        if nilj == 1:
            # using index on node1
            nilj_cost = card2 + NILJ_CONSTANT*card1
            # nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            # node1_selectivity = total1 / card1
            # joined_node_est = card3 * node1_selectivity
            # nilj_cost += joined_node_est
            cost = nilj_cost
        elif nilj == 2:
            # using index on node2
            nilj_cost = card1 + NILJ_CONSTANT*card2
            # nilj_cost = card1
            # node2_selectivity = total2 / card2
            # joined_node_est = card3 * node2_selectivity
            # nilj_cost += joined_node_est
            cost = nilj_cost
        elif nilj == 3:
            cost = card1*card2
        else:
            assert False

    elif cost_model == "nested_loop_index9":
        if nilj == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est

        elif nilj == 2:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        cost = nilj_cost

    elif cost_model == "nested_loop_index13":
        # same as nli7 --> but consider the fact the right side of an index
        # nested loop join WILL not have predicates pushed down
        # also, remove the term for index entirely
        if nilj == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est
            nilj_cost *= penalty

        elif nilj == 2:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
            nilj_cost *= penalty
        else:
            assert False

        # TODO: we may be doing fine without this one
        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop_index14":
        cost2 = 0.0
        if nilj == 1:
            # using index on node1
            nilj_cost = card2 + NILJ_CONSTANT*card1
            cost2 += NILJ_CONSTANT*card1
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est

        elif nilj == 2:
            # using index on node2
            nilj_cost = card1 + NILJ_CONSTANT*card2
            cost2 += NILJ_CONSTANT*card2
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        # TODO: we may be doing fine without this one
        cost2 += card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop":
        cost = card1*card2
    elif cost_model == "hash_join":
        cost = CARD_DIVIDER*card1 + CARD_DIVIDER*card2
    else:
        assert False
    return cost

def get_optimization_variables_jax(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2, edges_head,
        nilj, cost_model, G, Q, penalties):
    '''
    returns costs, and updates numpy arrays G, Q in place.
    '''
    ests = jp.exp((yhat+min_val)*(max_val-min_val))
    costs = jp.zeros(len(edges_cost_node1))
    ests = jp.maximum(ests, 1.0)

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs = jax.ops.index_update(costs, jax.ops.index[i], 1.0)
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = ests[node1]
        card2 = ests[node2]
        card3 = ests[edges_head[i]]
        total1 = totals[node1]
        total2 = totals[node2]
        penalty = penalties[i]
        cost = get_costs_jax(card1, card2, card3, nilj[i], cost_model, total1,
                total2, penalty)
        assert cost != 0.0
        costs = jax.ops.index_update(costs, jax.ops.index[i], cost)

    costs = 1 / costs
    return costs

def get_edge_costs3(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj):
    ests = jp.exp((yhat+min_val)*(max_val-min_val))
    costs = jp.zeros(len(edges_cost_node1))

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs = jax.ops.index_update(costs, jax.ops.index[i], 1.0)
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = ests[node1]
        card2 = ests[node2]
        if nilj[i] == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj[i] == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False
        cost = nilj_cost
        assert cost != 0.0
        costs = jax.ops.index_update(costs, jax.ops.index[i], cost)

    costs = 1 / costs;
    return costs

def get_edge_costs2(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj):
    '''
    @ests: cardinality estimates for each nodes (sorted by node_names)
    @totals: Total estimates for each node (sorted ...)
    @min_val, max_val, normalization_type

    @edges_cost_node1:
    @edges_cost_node2: these are single tables..
    '''
    start = time.time()
    ests = to_variable(np.zeros(len(yhat), dtype=np.float32),
            requires_grad=True).float()
    ests.requires_grad = True
    for i in range(len(yhat)):
        if normalization_type == "mscn":
            ests[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            ests[i] = yhat[i]*totals[i]
        else:
            assert False

    dgdxT = torch.zeros(len(ests), len(edges_cost_node1))
    # costs = to_variable(np.zeros(len(edges_cost_node1)), requires_grad=True).float()
    costs = torch.zeros(len(edges_cost_node1), requires_grad=True).float()

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs[i] = 1.0
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = torch.add(ests[node1], 1.0)
        card2 = torch.add(ests[node2], 1.0)
        hash_join_cost = card1 + card2
        if nilj[i] == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj[i] == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            nilj_cost = 10000000000
        cost = torch.min(hash_join_cost, nilj_cost)
        assert cost != 0.0
        costs[i] = cost

        if normalization_type is None:
            continue

        # time to compute gradients
        if normalization_type == "pg_total_selectivity":
            total1 = totals[node1]
            total2 = totals[node2]
            if hash_join_cost < nilj_cost:
                assert cost == hash_join_cost
                # - (a / (ax_1 + bx_2)**2)
                dgdxT[node1, i] = - (total1 / ((hash_join_cost)**2))
                # - (b / (ax_1 + bx_2)**2)
                dgdxT[node2, i] = - (total2 / ((hash_join_cost)**2))
            else:
                # index nested loop join
                assert cost == nilj_cost
                if nilj[i] == 1:
                    # - (a / (ax_1 + bx_2)**2)
                    num1 = total1*NILJ_CONSTANT
                    dgdxT[node1, i] = - (num1 / ((cost)**2))
                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[node2, i] = - (total2 / ((cost)**2))
                else:
                    # node 2
                    # - (a / (ax_1 + bx_2)**2)
                    num2 = total2*NILJ_CONSTANT
                    dgdxT[node1, i] = - (total1 / ((cost)**2))
                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[node2, i] = - (num2 / ((cost)**2))
        else:
            if hash_join_cost <= nilj_cost:
                assert cost == hash_join_cost
                # - (ae^{ax} / (e^{ax} + e^{ax2})**2)
                # e^{ax} is just card1
                dgdxT[node1, i] = - (max_val*card1 / ((hash_join_cost)**2))
                dgdxT[node2, i] = - (max_val*card2 / ((hash_join_cost)**2))

            else:
                # index nested loop join
                assert cost == nilj_cost
                if nilj[i]  == 1:
                    dgdxT[node1, i] = - (max_val*card1*NILJ_CONSTANT / ((cost)**2))
                    dgdxT[node2, i] = - (max_val*card2 / ((cost)**2))
                else:
                    # num2 = card2*NILJ_CONSTANT
                    dgdxT[node1, i] = - (max_val*card1 / ((cost)**2))
                    dgdxT[node2, i] = - (max_val*card2*NILJ_CONSTANT / ((cost)**2))

    # print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

def single_forward2(yhat, totals, edges_head, edges_tail, edges_cost_node1,
        edges_cost_node2, nilj, normalization_type, min_val, max_val,
        trueC_vec, final_node, cost_model, penalties):
    '''
    @yhat: NN outputs for nodes (sorted by nodes.sort())
    @totals: Total estimates for each node (sorted ...)
    @edges_head: len() == num edges. Each element is an index of the head node
    in that edge
    @edges_tail: ...
    ## which nodes determine the cost in each edge
    @edges_cost_node1:
    @edges_cost_node2: these are single tables..
    '''
    if DEBUG_JAX:
        costs_grad_fn = jacfwd(get_optimization_variables_jax, argnums=0)
        # costs_grad_fn = jacrev(get_optimization_variables_jax, argnums=0)
        jax_start = time.time()
        yhat = np.array(yhat)
        G = np.zeros((len(yhat),len(yhat)), dtype=np.float32)
        Q = np.zeros((len(edges_cost_node1),len(yhat)), dtype=np.float32)

        costs = get_optimization_variables_jax(yhat, totals, min_val, max_val,
                normalization_type, edges_cost_node1, edges_cost_node2, edges_head,
                nilj, cost_model, G, Q, penalties)
        print(costs.shape, np.max(costs))
        costs_grad = jacfwd(get_optimization_variables_jax, argnums=0)(yhat, totals, min_val,
                max_val, normalization_type, edges_cost_node1, edges_cost_node2,
                edges_head, nilj, cost_model, G, Q, penalties)
        costs_grad = costs_grad.T
        print(costs_grad.shape, np.max(costs_grad))

        print("jax stuff took: ", time.time()-jax_start)
        yhat = torch.from_numpy(yhat)

    start = time.time()
    est_cards = to_variable(np.zeros(len(yhat), dtype=np.float32)).float()
    for i in range(len(yhat)):
        if normalization_type == "mscn":
            est_cards[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            est_cards[i] = yhat[i]*totals[i]
        else:
            assert False

    start = time.time()
    predC2, dgdxT2, G2, Q2 = get_optimization_variables(est_cards, totals,
            min_val, max_val, normalization_type, edges_cost_node1,
            edges_cost_node2, nilj, edges_head, edges_tail, cost_model,
            penalties)

    if DEBUG_JAX:
        print("min max estimates: ", np.min(est_cards.detach().numpy()),
                np.max(est_cards.detach().numpy()))
        print("non jax stuff took: ", time.time()-start)
        predC = 1.0 / predC2
        print(np.allclose(predC, costs))
        print(np.min(predC), np.max(predC))
        print(np.min(costs), np.max(costs))

        if not np.allclose(predC, costs):
            print("costs not close!")
            pdb.set_trace()

        print(np.allclose(dgdxT2, costs_grad))
        if not np.allclose(dgdxT2, costs_grad):
            print("cost grads not close!")
            print("norm diff: ", np.linalg.norm(dgdxT2 - costs_grad))

        print(np.min(dgdxT2), np.max(dgdxT2))
        print(np.min(costs_grad), np.max(costs_grad))

        pdb.set_trace()

    Gv2 = np.zeros(len(totals))
    Gv2[final_node] = 1.0

    mat_start = time.time()
    Gv2 = to_variable(Gv2).float()
    predC2 = to_variable(predC2).float()
    dgdxT2 = to_variable(dgdxT2).float()
    G2 = to_variable(G2).float()
    invG = torch.inverse(G2)
    v = invG @ Gv2 # vshape: Nx1
    v = v.detach().numpy()

    # TODO: we don't even need to compute the loss here if we don't want to
    loss2 = np.zeros(1, dtype=np.float32)
    assert Q2.dtype == np.float32
    assert v.dtype == np.float32
    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().numpy()
    assert trueC_vec.dtype == np.float32
    # just computes the loss
    fl_cpp.get_qvtqv(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            trueC_vec.ctypes.data_as(c_void_p),
            loss2.ctypes.data_as(c_void_p)
            )

    # print("forward took: ", time.time()-start)
    return to_variable(loss2).float(), dgdxT2.detach(), invG.detach().numpy(), Q2, v

def single_backward(Q, invG,
        v, dgdxT, opt_flow_loss, trueC_vec,
        edges_head, edges_tail, normalize_flow_loss):

    start = time.time()

    assert Q.dtype == np.float32
    assert invG.dtype == np.float32

    QinvG2 = np.zeros((Q.shape[0], invG.shape[1]), dtype=np.float32)
    fl_cpp.get_qinvg(c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q.ctypes.data_as(c_void_p),
            invG.ctypes.data_as(c_void_p),
            QinvG2.ctypes.data_as(c_void_p))

    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().numpy()

    assert trueC_vec.dtype == np.float32
    assert Q.dtype == np.float32
    assert v.dtype == np.float32

    tqv_start = time.time()
    tQv = np.zeros(len(edges_head), dtype=np.float32)
    fl_cpp.get_tqv(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            trueC_vec.ctypes.data_as(c_void_p),
            c_int(2),
            tQv.ctypes.data_as(c_void_p)
            )
    # print("tqv computations took: ", time.time()-tqv_start)
    dCdg2 = np.zeros(tQv.shape, dtype=np.float32)

    dfdg = np.zeros((len(edges_head), len(edges_head)), dtype=np.float32)
    dfdg_start = time.time()
    num_threads = int(len(edges_head) / 400)
    num_threads = max(1, num_threads)
    num_threads = min(20, num_threads)
    fl_cpp.get_dfdg(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            QinvG2.ctypes.data_as(c_void_p),
            tQv.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            dfdg.ctypes.data_as(c_void_p),
            dCdg2.ctypes.data_as(c_void_p),
            c_int(num_threads))

    # mat_start = time.time()
    # slow matrix mul when we have too many edges
    # dCdg = dfdg @ tQv
    # print("mat took: ", time.time()-mat_start)

    yhat_grad = dgdxT @ to_variable(dCdg2).float()

    if normalize_flow_loss:
        yhat_grad /= opt_flow_loss

    return yhat_grad

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, subsetg_vectors,
            normalize_flow_loss,
            pool, cost_model):
        '''
        '''
        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        # torch.set_num_threads(1)
        yhat = yhat.detach()
        ctx.pool = pool
        ctx.normalize_flow_loss = normalize_flow_loss
        ctx.subsetg_vectors = subsetg_vectors
        assert len(subsetg_vectors[0][0]) == 8
        start = time.time()
        ctx.dgdxTs = []
        ctx.invGs = []
        ctx.Qs = []
        ctx.vs = []

        totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                edges_cost_node2, final_node, edge_penalties = ctx.subsetg_vectors[0][0]
        trueC_vec, opt_flow_loss = ctx.subsetg_vectors[0][1], \
                        ctx.subsetg_vectors[0][2]

        start = time.time()
        res = single_forward2(yhat, totals,
                edges_head, edges_tail, edges_cost_node1,
                edges_cost_node2,
                nilj,
                normalization_type,
                min_val, max_val,
                trueC_vec, final_node, cost_model, edge_penalties)

        loss = res[0]
        ctx.dgdxTs.append(res[1])
        ctx.invGs.append(res[2])
        ctx.Qs.append(res[3])
        ctx.vs.append(res[4])

        # if normalize_flow_loss == 1:
            # loss = loss / opt_flow_loss

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return gradients wrt preds, and bunch of Nones
        '''
        # torch.set_grad_enabled(False)
        # torch.set_num_threads(1)
        start = time.time()
        assert ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]

        _, edges_head, edges_tail, _, _, \
                _, _,_ = ctx.subsetg_vectors[0][0]
        trueC_vec, opt_cost = ctx.subsetg_vectors[0][1], \
                                ctx.subsetg_vectors[0][2]

        yhat_grad = single_backward(
                         ctx.Qs[0], ctx.invGs[0],
                         ctx.vs[0], ctx.dgdxTs[0],
                         opt_cost, trueC_vec,
                         edges_head,
                         edges_tail,
                         ctx.normalize_flow_loss)

        return yhat_grad,None,None,None,None,None,None,None,None,None

