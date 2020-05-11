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

system = platform.system()
if system == 'Linux':
    lib_file = "libflowloss.so"
else:
    lib_file = "libflowloss.dylib"

lib_dir = "./flow_loss_cpp"
lib_file = lib_dir + "/" + lib_file
fl_cpp = CDLL(lib_file, mode=RTLD_GLOBAL)

DEBUG = False

def get_optimization_variables(ests, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj, edges_head, edges_tail):
    '''
    @ests: these are actual values for each estimate. totals,min_val,max_val
    are only required for the derivatives.
    '''
    start = time.time()

    # TODO: speed up this init stuff?
    if normalization_type == "mscn":
        norm_type = 2
    else:
        norm_type = 1
    # TODO: make sure everything is the correct type beforehand
    if min_val is None:
        min_val = 0.0
        max_val = 0.0

    if not isinstance(ests, np.ndarray):
        ests = ests.detach().numpy()

    # costs = np.zeros(len(edges_cost_node1), dtype=np.float32)
    # dgdxT = np.zeros((len(ests), len(edges_cost_node1)), dtype=np.float32)
    # G = np.zeros((len(ests),len(ests)), dtype=np.float32)
    # Q = np.zeros((len(costs),len(ests)), dtype=np.float32)

    costs2 = np.zeros(len(edges_cost_node1), dtype=np.float32)
    dgdxT2 = np.zeros((len(ests), len(edges_cost_node1)), dtype=np.float32)
    G2 = np.zeros((len(ests),len(ests)), dtype=np.float32)
    Q2 = np.zeros((len(edges_cost_node1),len(ests)), dtype=np.float32)

    start = time.time()
    fl_cpp.get_optimization_variables(ests.ctypes.data_as(c_void_p),
            totals.ctypes.data_as(c_void_p),
            c_double(min_val),
            c_double(max_val),
            c_int(norm_type),
            edges_cost_node1.ctypes.data_as(c_void_p),
            edges_cost_node2.ctypes.data_as(c_void_p),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            nilj.ctypes.data_as(c_void_p),
            c_int(len(ests)),
            c_int(len(costs2)),
            costs2.ctypes.data_as(c_void_p),
            dgdxT2.ctypes.data_as(c_void_p),
            G2.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p))

    return costs2, dgdxT2, G2, Q2


def get_edge_costs2(ests, totals, min_val, max_val,
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
    dgdxT = torch.zeros(len(ests), len(edges_cost_node1))
    costs = to_variable(np.zeros(len(edges_cost_node1))).float()

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
        trueC_vec, final_node):
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
            edges_cost_node2, nilj, edges_head, edges_tail)
    # print("get opt variables took: ", time.time()-start)

    Gv2 = np.zeros(len(totals))
    Gv2[final_node] = 1.0

    Gv2 = to_variable(Gv2).float()
    predC2 = to_variable(predC2).float()
    dgdxT2 = to_variable(dgdxT2).float()
    G2 = to_variable(G2).float()
    invG = torch.inverse(G2)
    v = invG @ Gv2 # vshape: Nx1
    v = v.detach().numpy()

    # TODO: we don't even need to compute the loss here if we don't want to
    mat_start = time.time()
    loss2 = np.zeros(1, dtype=np.float32)
    assert Q2.dtype == np.float32
    assert v.dtype == np.float32
    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().numpy()
    assert trueC_vec.dtype == np.float32
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

    dfdg = np.zeros((len(edges_head), len(edges_head)), dtype=np.float32)
    dfdg_start = time.time()
    num_threads = int(len(edges_head) / 400)
    num_threads = max(1, num_threads)
    fl_cpp.get_dfdg(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            QinvG2.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            dfdg.ctypes.data_as(c_void_p),
            c_int(num_threads))
    # print("dfdg took: ", time.time()-dfdg_start)

    dfdg = to_variable(dfdg).float()
    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().numpy()

    assert trueC_vec.dtype == np.float32
    assert Q.dtype == np.float32
    assert v.dtype == np.float32

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

    dCdg = dfdg @ tQv

    yhat_grad = dgdxT @ dCdg
    if normalize_flow_loss:
        yhat_grad /= opt_flow_loss

    return yhat_grad

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, subsetg_vectors,
            normalize_flow_loss,
            pool):
        '''
        '''
        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        # torch.set_num_threads(1)
        yhat = yhat.detach()
        ctx.pool = pool
        ctx.normalize_flow_loss = normalize_flow_loss
        ctx.subsetg_vectors = subsetg_vectors
        assert len(subsetg_vectors[0]) == 9
        start = time.time()
        ctx.dgdxTs = []
        ctx.invGs = []
        ctx.Qs = []
        ctx.vs = []

        if len(subsetg_vectors) > 1:
            # FIXME: later
            assert False
            par_args = []
            qidx = 0
            for i, subsetg in enumerate(subsetgs):
                # num nodes = num of yhat predictions
                num_nodes = len(subsetg.nodes())-1
                par_args.append((yhat[qidx:qidx+num_nodes],
                                 normalization_type,
                                 min_val,
                                 max_val,
                                 node_dicts[i],
                                 edge_dicts[i],
                                 subsetg,
                                 # trueCs[i].detach(),
                                 final_nodes[i]))
                qidx += num_nodes

            results = ctx.pool.starmap(single_forward, par_args)
            loss = 0.0
            for i, res in enumerate(results):
                loss += res[0]
                ctx.dgdxTs.append(res[1])
                ctx.invGs.append(res[2])
                ctx.Qs.append(res[3])
                ctx.vs.append(res[4])
        else:
            totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                    edges_cost_node2, final_node, trueC_vec, opt_cost \
                        = ctx.subsetg_vectors[0]

            start = time.time()
            res = single_forward2(yhat, totals,
                    edges_head, edges_tail, edges_cost_node1,
                    edges_cost_node2,
                    nilj,
                    normalization_type,
                    min_val, max_val,
                    trueC_vec, final_node)

            loss = res[0]
            ctx.dgdxTs.append(res[1])
            ctx.invGs.append(res[2])
            ctx.Qs.append(res[3])
            ctx.vs.append(res[4])

        # print("forward took: ", time.time()-start)
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

        if len(ctx.subsetg_vectors) > 1:
            par_args = []
            for i in range(len(ctx.subsetgs)):
                par_args.append((ctx.edge_dicts[i],
                                 ctx.node_dicts[i],
                                 ctx.Qs[i], ctx.invGs[i],
                                 ctx.vs[i], ctx.dgdxTs[i],
                                 ctx.opt_flow_losses[i], ctx.trueCs[i]))

            results = ctx.pool.starmap(single_backward, par_args)
            yhat_grad = np.concatenate(results)
            yhat_grad /= len(ctx.subsetgs)
            yhat_grad = torch.from_numpy(yhat_grad)
        else:

            _, edges_head, edges_tail, _, _, \
                    _, _, trueC_vec, opt_cost \
                        = ctx.subsetg_vectors[0]
            yhat_grad = single_backward(
                             ctx.Qs[0], ctx.invGs[0],
                             ctx.vs[0], ctx.dgdxTs[0],
                             opt_cost, trueC_vec,
                             edges_head,
                             edges_tail,
                             ctx.normalize_flow_loss)

        return yhat_grad,None,None,None,None,None,None,None,None

