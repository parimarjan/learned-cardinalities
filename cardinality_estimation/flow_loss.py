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

def compute_dfdg_row_np(edge_num, edge, node_dict, QinvG,
        v):
    head_node = node_dict[edge[0]]
    if edge[1] in node_dict:
        tail_node = node_dict[edge[1]]
        dGdgi = np.ones((2,2))
        dGdgi[0,1] = -1.0
        dGdgi[1,0] = -1.0

        QinvGSlice = np.zeros((QinvG.shape[0], 2))
        QinvGSlice[:,0] = QinvG[:,head_node]
        QinvGSlice[:,1] = QinvG[:,tail_node]
        # QinvGSlice = QinvG[:,[head_node, tail_node]]
        rightSlice = - (QinvGSlice @ dGdgi)

        # add dQ/dgi's relevant valeus
        rightSlice[edge_num, 0] += 1.0
        rightSlice[edge_num, 1] -= 1.0
        vT = np.zeros(2)
        vT[0] = v[head_node]
        vT[1] = v[tail_node]
    else:
        # dGdgi = np.ones((1,1))
        # dGdgi[0,0] = 1.0
        QinvGSlice = -QinvG[:, head_node]
        rightSlice = np.reshape(QinvGSlice, (QinvG.shape[0], 1))
        # rightSlice = - (QinvGSlice * dGdgi)

        rightSlice[edge_num, 0] += 1.0
        vT = v[head_node]

    vT = np.reshape(vT, (1, rightSlice.shape[1]))
    ret = vT @ rightSlice.T
    return ret

def compute_dfdg_row(edge_num, edge, node_dict, QinvG, v):
    head_node = node_dict[edge[0]]
    if edge[1] in node_dict:
        tail_node = node_dict[edge[1]]
        dGdgi = torch.ones((2,2))
        dGdgi[0,1] = -1.0
        dGdgi[1,0] = -1.0

        QinvGSlice = torch.zeros((QinvG.shape[0], 2))
        QinvGSlice[:,0] = QinvG[:,head_node]
        QinvGSlice[:,1] = QinvG[:,tail_node]
        rightSlice = - (QinvGSlice @ dGdgi)

        # add dQ/dgi's relevant valeus
        rightSlice[edge_num, 0] += 1.0
        rightSlice[edge_num, 1] -= 1.0
        vT = torch.zeros(2)
        vT[0] = v[head_node]
        vT[1] = v[tail_node]
    else:
        dGdgi = torch.ones((1,1))
        dGdgi[0,0] = 1.0
        QinvGSlice = QinvG[:, head_node]
        QinvGSlice = torch.reshape(QinvGSlice, (QinvG.shape[0], 1))
        rightSlice = - (QinvGSlice * dGdgi)

        rightSlice[edge_num, 0] += 1.0
        vT = v[head_node]

    vT = torch.reshape(vT, (1, rightSlice.shape[1]))
    ret = vT @ torch.transpose(rightSlice, 0, 1)
    return ret

def constructG(subsetg, preds, node_dict, edge_dict,
        final_node):
    '''
    TODO:
        sorted list of nodes, edges, node_dict, edge_dict will be args
            + final_node
    '''
    start = time.time()
    N = len(subsetg.nodes()) - 1
    M = len(subsetg.edges())
    G = to_variable(np.zeros((N,N))).float()
    Q = to_variable(np.zeros((M,N))).float()
    Gv = to_variable(np.zeros(N)).float()
    Gv[node_dict[final_node]] = 1.0

    # FIXME: this loop is surprisingly expensive, can we convert it to matrix ops?
    for edge, i in edge_dict.items():
        cost = preds[i]
        cost = 1.0 / cost

        head_node = edge[0]
        tail_node = edge[1]
        hidx = node_dict[head_node]
        Q[i,hidx] = cost
        G[hidx,hidx] += cost

        if tail_node in node_dict:
            tidx = node_dict[tail_node]
            Q[i,tidx] = -cost
            G[tidx,tidx] += cost
            G[hidx,tidx] -= cost
            G[tidx,hidx] -= cost

    return G, Gv, Q

def get_optimization_variables(ests, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj, edges_head, edges_tail):
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

    # TODO: this should be in the correct format already...
    ests = ests.detach().numpy()
    # edges_cost_node1 = np.array(edges_cost_node1, dtype=np.int32)
    # edges_cost_node2 = np.array(edges_cost_node2, dtype=np.int32)
    # edges_head = np.array(edges_head, dtype=np.int32)
    # edges_tail = np.array(edges_tail, dtype=np.int32)
    # nilj = np.array(nilj, dtype=np.int32)

    costs = np.zeros(len(edges_cost_node1), dtype=np.float32)
    dgdxT = np.zeros((len(ests), len(edges_cost_node1)), dtype=np.float32)
    G = np.zeros((len(ests),len(ests)), dtype=np.float32)
    Q = np.zeros((len(costs),len(ests)), dtype=np.float32)

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
            c_int(len(costs)),
            costs.ctypes.data_as(c_void_p),
            dgdxT.ctypes.data_as(c_void_p),
            G.ctypes.data_as(c_void_p),
            Q.ctypes.data_as(c_void_p))

    # print("get optimization variables took: ", time.time()-start)
    return costs, dgdxT, G, Q


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

    print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

def get_edge_costs(subsetg, ests, node_dict, edge_dict,
        normalization_type, min_val, max_val):
    '''
    @ret: returns costs for each edge in the subset graph.
    '''
    start = time.time()
    dgdxT = torch.zeros(len(node_dict), len(edge_dict))

    costs = to_variable(np.zeros(len(edge_dict))).float()

    for edge, edgei in edge_dict.items():
        # TODO: need a different way to specify the source edge
        if len(edge[0]) == len(edge[1]):
            assert edge[1] == SOURCE_NODE
            costs[edgei] = 1.0
            continue

        # FIXME: how do we get information about this from arrays?
        assert len(edge[1]) < len(edge[0])
        assert edge[1][0] in edge[0]
        ## FIXME:
        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        # node2.sort()
        node2 = tuple(node2)
        assert node2 in subsetg.nodes()

        # FIXME: this dictionary is ONLY used for totals, waste
        cards1 = subsetg.nodes()[node1]["cardinality"]
        cards2 = subsetg.nodes()[node2]["cardinality"]

        idx1 = node_dict[node1]
        idx2 = node_dict[node2]
        card1 = torch.add(ests[node_dict[node1]], 1.0)
        card2 = torch.add(ests[node_dict[node2]], 1.0)

        # assert card1 >= 0.90
        # if card2 < 0.90:
            # print(card2)
            # pdb.set_trace()

        hash_join_cost = card1 + card2
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            nilj_cost = 10000000000
        cost = torch.min(hash_join_cost, nilj_cost)
        assert cost != 0.0
        costs[edgei] = cost

        if normalization_type is None:
            continue

        # time to compute gradients
        if normalization_type == "pg_total_selectivity":
            total1 = cards1["total"]
            total2 = cards2["total"]
            if hash_join_cost < nilj_cost:
                assert cost == hash_join_cost
                # - (a / (ax_1 + bx_2)**2)
                dgdxT[idx1, edgei] = - (total1 / ((hash_join_cost)**2))

                # - (b / (ax_1 + bx_2)**2)
                dgdxT[idx2, edgei] = - (total2 / ((hash_join_cost)**2))

            else:
                # index nested loop join
                assert cost == nilj_cost
                if len(node1) == 1:
                    # - (a / (ax_1 + bx_2)**2)
                    num1 = total1*NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (num1 / ((cost)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, edgei] = - (total2 / ((cost)**2))

                else:
                    # node 2
                    # - (a / (ax_1 + bx_2)**2)
                    num2 = total2*NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (total1 / ((cost)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, edgei] = - (num2 / ((cost)**2))
        else:
            if hash_join_cost <= nilj_cost:
                assert cost == hash_join_cost
                # - (ae^{ax} / (e^{ax} + e^{ax2})**2)
                # e^{ax} is just card1
                dgdxT[idx1, edgei] = - (max_val*card1 / ((hash_join_cost)**2))

                dgdxT[idx2, edgei] = - (max_val*card2 / ((hash_join_cost)**2))

            else:
                # index nested loop join
                assert cost == nilj_cost
                if len(node1) == 1:
                    dgdxT[idx1, edgei] = - (max_val*card1*NILJ_CONSTANT / ((cost)**2))
                    dgdxT[idx2, edgei] = - (max_val*card2 / ((cost)**2))

                else:
                    # num2 = card2*NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (max_val*card1 / ((cost)**2))
                    dgdxT[idx2, edgei] = - (max_val*card2*NILJ_CONSTANT / ((cost)**2))

    print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

def single_forward2(yhat, totals, edges_head, edges_tail, edges_cost_node1,
        edges_cost_node2, nilj,
        normalization_type, min_val, max_val, trueC,
        final_node):
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
    # totals = np.zeros(len(yhat), dtype=np.float32)
    # edges_head = [0]*len(edge_dict)
    # edges_tail = [0]*len(edge_dict)
    # edges_cost_node1 = [0]*len(edge_dict)
    # edges_cost_node2 = [0]*len(edge_dict)
    # nilj = [0]*len(edge_dict)

    # for node, nodei in node_dict.items():
        # totals[nodei] = subsetg.nodes()[node]["cardinality"]["total"]

    # for edge, edgei in edge_dict.items():
        # if len(edge[0]) == len(edge[1]):
            # assert edge[1] == SOURCE_NODE
            # edges_head[edgei] = node_dict[edge[0]]
            # edges_tail[edgei] = SOURCE_NODE_CONST
            # edges_cost_node1[edgei] = SOURCE_NODE_CONST
            # edges_cost_node2[edgei] = SOURCE_NODE_CONST
            # continue

        # edges_head[edgei] = node_dict[edge[0]]
        # edges_tail[edgei] = node_dict[edge[1]]

        # # FIXME: how do we get information about this from arrays?
        # assert len(edge[1]) < len(edge[0])
        # assert edge[1][0] in edge[0]
        # ## FIXME:
        # node1 = edge[1]
        # diff = set(edge[0]) - set(edge[1])
        # node2 = list(diff)
        # # node2.sort()
        # node2 = tuple(node2)
        # assert node2 in subsetg.nodes()

        # edges_cost_node1[edgei] = node_dict[node1]
        # edges_cost_node2[edgei] = node_dict[node2]

        # if len(node1) == 1:
            # # nilj_cost = card2 + NILJ_CONSTANT*card1
            # nilj[edgei] = 1
        # elif len(node2) == 1:
            # nilj[edgei] = 2

    est_cards = to_variable(np.zeros(len(yhat))).float()
    for i in range(len(yhat)):
        if normalization_type == "mscn":
            est_cards[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            est_cards[i] = yhat[i]*totals[i]
        else:
            assert False


    predC2, dgdxT2, G2, Q2 = get_optimization_variables(est_cards, totals,
            min_val, max_val, normalization_type, edges_cost_node1,
            edges_cost_node2, nilj, edges_head, edges_tail)
    Gv2 = to_variable(np.zeros(len(totals))).float()
    Gv2[final_node] = 1.0

    ## debug code
    # predC, dgdxT = get_edge_costs(subsetg, est_cards, node_dict, edge_dict,
            # normalization_type, min_val, max_val)
    # G,Gv,Q = constructG(subsetg, predC, node_dict, edge_dict, final_node)
    # assert np.allclose(predC, predC2)
    # assert np.allclose(dgdxT.detach().numpy(), dgdxT2)
    # if not np.allclose(G.detach().numpy(), G2):
        # print(np.linalg.norm(G.detach().numpy()-G2))
        # print(np.linalg.norm(Q.detach().numpy()-Q2))
        # print("G not same!")
        # pdb.set_trace()
    # assert np.allclose(Q.detach().numpy(), Q2)
    # pdb.set_trace()

    predC2 = to_variable(predC2).float()
    dgdxT2 = to_variable(dgdxT2).float()
    G2 = to_variable(G2).float()
    Q2 = to_variable(Q2).float()

    mat_start = time.time()
    invG = torch.inverse(G2)
    v = invG @ Gv2
    left = (Gv2 @ torch.transpose(invG,0,1)) @ torch.transpose(Q2, 0, 1)
    right = Q2 @ (v)
    loss = left @ trueC @ right

    return loss, dgdxT2.detach(), invG.detach(), Q2.detach(), v.detach()

def single_forward(yhat, normalization_type, min_val,
        max_val, node_dict, edge_dict, subsetg, trueC,
        final_node):
    '''
    '''
    # torch.set_grad_enabled(False)
    est_cards = to_variable(np.zeros(len(yhat))).float()

    # TODO: what is sorted order of each of the arrays?
    # TODO: can just replace this with loop over sorted node array
    # access each i in sorted manner...
    # node name is ONLY used to access total, totals should be an appropriately
    # sorted array already
    ## Should NOT NEED node names ever.

    for node,i in node_dict.items():
        if normalization_type == "mscn":
            est_cards[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            est_cards[i] = yhat[i]*subsetg.nodes()[node]["cardinality"]["total"]
        else:
            assert False

    # TODO: simplify get_edge_costs further
    predC, dgdxT = get_edge_costs(subsetg, est_cards, node_dict, edge_dict,
            normalization_type, min_val, max_val)

    # calculate flow loss
    G,Gv,Q = constructG(subsetg, predC, node_dict, edge_dict, final_node)

    mat_start = time.time()
    invG = torch.inverse(G)
    v = invG @ Gv
    left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
    right = Q @ (v)
    loss = left @ trueC @ right

    return loss, dgdxT.detach(), invG.detach(), Q.detach(), v.detach()

def single_backward(Q, invG,
        v, dgdxT, opt_flow_loss, trueC,
        edges_head, edges_tail, normalize_flow_loss):

    start = time.time()
    QinvG = Q @ invG
    QinvG = QinvG.detach().numpy()
    v = v.detach().numpy()

    # dfdg_old = torch.zeros((trueC.shape))
    # compute each row
    # rstart = time.time()

    # for edge, idx in edge_dict.items():
        # # corresponding to ith cost edge
        # # row = compute_dfdg_row_np(idx, edge, node_dict, QinvG,
                # # v)
        # # dfdg_old[idx,:] = torch.from_numpy(row)
        # row = compute_dfdg_row(idx, edge, node_dict,
                # to_variable(QinvG).float(),
                # to_variable(v).float())
        # dfdg_old[idx,:] = row

    # print("computing dfdg took: ", time.time()-rstart)

    dfdg = np.zeros((trueC.shape), dtype=np.float32)
    rstart2 = time.time()

    fl_cpp.get_dfdg(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            QinvG.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            dfdg.ctypes.data_as(c_void_p),
            c_int(20))

    # print("computing dfdg2 took: ", time.time()-rstart2)

    ## debug code
    # print(dfdg[0,0], dfdg[0,1])
    # print(dfdg_old[0,0], dfdg_old[0,1])
    # print("norm diff: ", np.linalg.norm(dfdg - dfdg_old.detach().numpy()))
    # print(np.allclose(dfdg_old.detach().numpy(), dfdg))
    # pdb.set_trace()

    dfdg = to_variable(dfdg).float()
    dCdg = 2 * (dfdg @ (trueC @ (Q @ v)))

    yhat_grad = dgdxT @ dCdg
    if normalize_flow_loss:
        yhat_grad /= opt_flow_loss
    return yhat_grad

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, subsetg_vectors,
            normalize_flow_loss,
            # node_dicts, edge_dicts, subsetgs,
            # trueCs, opt_flow_losses, final_nodes, con_mats,
            pool):
        '''
        '''
        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        yhat = yhat.detach()
        # ctx.yhat = yhat
        ctx.pool = pool
        ctx.normalize_flow_loss = normalize_flow_loss
        # subsetg_vectors ret: totals, edges_head, edges_tail, nilj, \
                # edges_cost_node1, edges_cost_node2, final_node, trueC,
                # opt_cost
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
                                 trueCs[i].detach(),
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
            # subsetg_vectors ret: totals, edges_head, edges_tail, nilj, \
                    # edges_cost_node1, edges_cost_node2, final_node, trueC,
                    # opt_cost
            totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                    edges_cost_node2, final_node, trueC, opt_cost \
                        = ctx.subsetg_vectors[0]

            # pdb.set_trace()
            start = time.time()
            res = single_forward2(yhat, totals,
                    edges_head, edges_tail, edges_cost_node1,
                    edges_cost_node2,
                    nilj,
                    normalization_type,
                    min_val, max_val,
                    trueC.detach(), final_node)
            loss = res[0]
            ctx.dgdxTs.append(res[1])
            ctx.invGs.append(res[2])
            ctx.Qs.append(res[3])
            ctx.vs.append(res[4])

            ## debug code:
            # print("single forward done!")
            # res = single_forward(yhat, normalization_type,
                    # min_val, max_val, node_dicts[0],
                    # edge_dicts[0], subsetgs[0],
                    # trueCs[0].detach(), final_nodes[0])
            # for i,r in enumerate(res):
                # print(i, np.allclose(res2[i].detach().numpy(),
                    # r.detach().numpy()))
            # pdb.set_trace()

        # save for backward
        # ctx.edge_dicts = edge_dicts
        # ctx.node_dicts = node_dicts
        # ctx.subsetgs = subsetgs
        # ctx.normalization_type = normalization_type
        # ctx.min_val = min_val
        # ctx.max_val = max_val
        # ctx.opt_flow_losses = opt_flow_losses
        # ctx.trueCs = trueCs

        # print("forward took: ", time.time()-start)
        # print("loss: ", loss)
        # pdb.set_trace()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return gradients wrt preds, and bunch of Nones
        '''
        # torch.set_grad_enabled(False)
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
                    _, _, trueC, opt_cost \
                        = ctx.subsetg_vectors[0]
            yhat_grad = single_backward(
                             # ctx.edge_dicts[0],
                             # ctx.node_dicts[0],
                             ctx.Qs[0], ctx.invGs[0],
                             ctx.vs[0], ctx.dgdxTs[0],
                             opt_cost, trueC,
                             edges_head,
                             edges_tail,
                             ctx.normalize_flow_loss)

        # print("backward took: ", time.time()-start)
        # pdb.set_trace()

        return yhat_grad,None,None,None,None,None,None,None,None

