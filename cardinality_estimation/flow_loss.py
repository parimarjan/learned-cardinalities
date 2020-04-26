from torch.autograd import Function
from utils.utils import *
from db_utils.utils import *
import torch
import time
import pdb
from multiprocessing import Pool
from scipy import sparse

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

def compute_dfdg_row(edge_num, edge, node_dict, QinvG, v,
        subsetg):
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

def constructG2(subsetg, preds, node_dict, edge_dict,
        final_node, con_mat):
    '''
    TODO:
        sorted list of nodes, edges, node_dict, edge_dict will be args
            + final_node
    '''
    con_mat = con_mat.detach().numpy()
    con_mat = sparse.csr_matrix(con_mat)
    start = time.time()
    # Gv can be pre-computed too
    N = len(subsetg.nodes()) - 1
    Gv = to_variable(np.zeros(N)).float()
    Gv[node_dict[final_node]] = 1.0

    preds = 1.0 / preds
    diagC = np.diag(preds)

    # TODO: explain
    Q = diagC @ con_mat
    G = Q.T @ con_mat

    # print("constructG took: ", time.time()-start)
    return G, Gv, Q

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

def get_edge_costs(subsetg, ests, node_dict, edge_dict,
        normalization_type, min_val, max_val):
    '''
    @ret: returns costs for each edge in the subset graph.
    '''
    start = time.time()
    dgdxT = torch.zeros(len(node_dict), len(edge_dict))

    costs = to_variable(np.zeros(len(edge_dict))).float()
    for edge, edgei in edge_dict.items():

        if len(edge[0]) == len(edge[1]):
            assert edge[1] == SOURCE_NODE
            costs[edgei] = 1.0
            continue

        assert len(edge[1]) < len(edge[0])
        assert edge[1][0] in edge[0]
        ## FIXME:
        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        # node2.sort()
        node2 = tuple(node2)
        assert node2 in subsetg.nodes()

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
                # - (e^{ax} / (e^{ax} + e^{ax2})**2)
                # e^{ax} is just card1
                dgdxT[idx1, edgei] = - (card1 / ((hash_join_cost)**2))

                # - (b / (ax_1 + bx_2)**2)
                dgdxT[idx2, edgei] = - (card2 / ((hash_join_cost)**2))

            else:
                # index nested loop join
                assert cost == nilj_cost
                if len(node1) == 1:
                    num1 = card1*NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (num1 / ((cost)**2))
                    dgdxT[idx2, edgei] = - (card2 / ((cost)**2))

                else:
                    num2 = card2*NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (card1 / ((cost)**2))
                    dgdxT[idx2, edgei] = - (num2 / ((cost)**2))

    # print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

def single_forward(yhat, normalization_type, min_val,
        max_val, node_dict, edge_dict, subsetg, trueC,
        final_node):
    est_cards = to_variable(np.zeros(len(yhat))).float()
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
    # ctx.dgdxT = dgdxT

    # calculate flow loss
    G,Gv,Q = constructG(subsetg, predC, node_dict, edge_dict, final_node)

    mat_start = time.time()
    invG = torch.inverse(G)
    v = invG @ Gv
    left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
    right = Q @ (v)
    loss = left @ trueC @ right
    return loss, dgdxT.detach(), invG.detach(), Q.detach(), v.detach()

def single_backward(subsetg, edge_dict, node_dict, Q, invG,
        v, dgdxT, opt_flow_loss, trueC):
    start = time.time()
    N = len(subsetg.nodes()) - 1
    M = len(subsetg.edges())
    dfdg = torch.zeros((M,M))

    # compute each row
    rstart = time.time()
    QinvG = Q @ invG
    QinvG = QinvG.detach().numpy()
    v = v.detach().numpy()

    ## original
    for edge, idx in edge_dict.items():
        # corresponding to ith cost edge
        row = compute_dfdg_row_np(idx, edge, node_dict, QinvG,
                v)
        dfdg[idx,:] = torch.from_numpy(row)

    # print("computing dfdg took: ", time.time()-rstart)

    dCdg = 2 * (dfdg @ (trueC @ (Q @ v)))

    # update_list("dCdg_grad.pkl", dCdg.detach().numpy())
    # print("backwards, dCdg: ", dCdg)

    yhat_grad = dgdxT @ dCdg
    # print("single backward took: ", time.time()-start)
    yhat_grad /= opt_flow_loss
    return yhat_grad.detach().numpy()

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, node_dicts, edge_dicts, subsetgs,
            trueCs, opt_flow_losses, final_nodes, con_mats,
            pool):
        '''
        '''
        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        yhat = yhat.detach()
        ctx.pool = pool
        start = time.time()
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

        # print("single forward took: ", time.time()-start)

        # return loss, dgdxT, invG, Q, v
        loss = 0.0
        ctx.dgdxTs = []
        ctx.invGs = []
        ctx.Qs = []
        ctx.vs = []
        for i, res in enumerate(results):
            loss += res[0]
            ctx.dgdxTs.append(res[1])
            ctx.invGs.append(res[2])
            ctx.Qs.append(res[3])
            ctx.vs.append(res[4])

        # save for backward
        ctx.edge_dicts = edge_dicts
        ctx.node_dicts = node_dicts
        ctx.subsetgs = subsetgs
        ctx.normalization_type = normalization_type
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.opt_flow_losses = opt_flow_losses
        ctx.trueCs = trueCs
        return loss / len(subsetgs)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return gradients wrt preds, and bunch of Nones
        '''
        start = time.time()
        assert ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]

        par_args = []
        for i in range(len(ctx.subsetgs)):
            par_args.append((ctx.subsetgs[i],
                             ctx.edge_dicts[i],
                             ctx.node_dicts[i],
                             ctx.Qs[i], ctx.invGs[i],
                             ctx.vs[i], ctx.dgdxTs[i],
                             ctx.opt_flow_losses[i], ctx.trueCs[i]))

        results = ctx.pool.starmap(single_backward, par_args)
        yhat_grad = np.concatenate(results)
        yhat_grad /= len(ctx.subsetgs)

        return torch.from_numpy(yhat_grad),None, None, None, None, None, \
                            None,None,None,None,None,None,None
