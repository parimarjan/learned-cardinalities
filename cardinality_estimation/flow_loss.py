from torch.autograd import Function
from utils.utils import *
from db_utils.utils import *
import torch
import time
import pdb
from multiprocessing import Pool

DEBUG = False

def compute_dgdxT(yhat, subsetg, node_dict, edges, edge_dict,
        normalization_type, min_val, max_val):
    start = time.time()
    dgdxT = torch.zeros(len(node_dict), len(edges))

    for i, edge in enumerate(edges):
        assert i == edge_dict[edge]
        # which two nodes were involved in this
        if len(edge[0]) == len(edge[1]):
            assert edge[1] == SOURCE_NODE
            # derivatives would be 0 here so ignore this
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
        pred1 = yhat[idx1]
        pred2 = yhat[idx2]

        if normalization_type == "pg_total_selectivity":
            total1 = cards1["total"]
            total2 = cards2["total"]
            card1 = pred1 * total1
            card2 = pred2 * total2
            # need to decide between hash and merge join
            hash_join_cost = card1 + card2
            if len(node1) == 1:
                nilj_cost = card2 + NILJ_CONSTANT*card1
            elif len(node2) == 1:
                nilj_cost = card1 + NILJ_CONSTANT*card2
            else:
                nilj_cost = 10000000000
            # cost = torch.min(hash_join_cost, nilj_cost)
            if hash_join_cost < nilj_cost:
                # hash join
                cost = hash_join_cost
                # - (a / (ax_1 + bx_2)**2)
                dgdxT[idx1, i] = - (total1 / ((card1 + card2)**2))

                # - (b / (ax_1 + bx_2)**2)
                dgdxT[idx2, i] = - (total2 / ((card1 + card2)**2))

            else:
                # index nested loop join
                cost = nilj_cost
                if len(node1) == 1:
                    # - (a / (ax_1 + bx_2)**2)
                    total1 *= NILJ_CONSTANT
                    dgdxT[idx1, i] = - (total1 / ((NILJ_CONSTANT*card1 + card2)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, i] = - (total2 / ((NILJ_CONSTANT*card1 + card2)**2))

                else:
                    # node 2
                    # - (a / (ax_1 + bx_2)**2)
                    total2 *= NILJ_CONSTANT
                    dgdxT[idx1, i] = - (total1 / ((card1 + NILJ_CONSTANT*card2)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, i] = - (total2 / ((card1 + NILJ_CONSTANT*card2)**2))
        else:
            assert False

        # will update the values: [i, idx1] and [i, idx2], which will be the
        # partial derivative of the ith cost edge, with the two nodes on this
        # edge

    # print("dgdxT took: ", time.time()-start)
    return dgdxT

def compute_dfdg_row_np(edge_num, edge, node_dict, QinvG, v,
        subsetg):

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

def compute_dfdg_row_orig(edge_num, edge, node_dict, Q, G, invG, v,
        subsetg):

    dGdgi = torch.zeros(G.shape)
    dQdgi = torch.zeros(Q.shape)

    head_node = node_dict[edge[0]]
    dGdgi[head_node, head_node] = 1.0

    dQdgi[edge_num, head_node] = 1.0

    if edge[1] in node_dict:
        tail_node = node_dict[edge[1]]
        dGdgi[head_node, tail_node] = -1.0
        dGdgi[tail_node, tail_node] = +1.0
        dGdgi[tail_node, head_node] = -1.0

        dQdgi[edge_num, tail_node] = -1.0
    else:
        # don't need to adjust any value
        pass

    # now we've all the ingredients for the formula
    vT = v

    ## simplification:
    # invG_col = invG[:,head_node]
    # invG_col = torch.reshape(invG_col, (len(invG_col), 1))
    # dGdgi_row = dGdgi[head_node,:]
    # dGdgi_row = torch.reshape(dGdgi_row, (1, len(dGdgi_row)))
    # simpleRight = invG_col @ dGdgi_row
    # QinvGdGdgi = Q @ simpleRight

    # print(simpleRight.shape)
    # pdb.set_trace()

    ## without simplification:
    QinvGdGdgi = Q @ invG @ dGdgi

    right = torch.transpose(dQdgi - QinvGdGdgi, 0, 1)
    ret = vT @ right
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

    # print("constructG took: ", time.time()-start)
    return G, Gv, Q

def get_edge_costs(subsetg, ests, node_dict, edge_dict,
        normalization_type, min_val,
        max_val):
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
        # pred1 = yhat[idx1]
        # pred2 = yhat[idx2]
        card1 = ests[node_dict[node1]]
        card2 = ests[node_dict[node2]]

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
                    total1 *= NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (total1 / ((cost)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, edgei] = - (total2 / ((cost)**2))

                else:
                    # node 2
                    # - (a / (ax_1 + bx_2)**2)
                    total2 *= NILJ_CONSTANT
                    dgdxT[idx1, edgei] = - (total1 / ((cost)**2))

                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[idx2, edgei] = - (total2 / ((cost)**2))

        else:
            assert False

    # print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, node_dict, edge_dict, subsetg,
            trueC, opt_flow_loss, final_node):
        '''
        '''
        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        # ctx.subset_graph = sample["subset_graph"]
        start = time.time()
        ctx.subsetg = subsetg
        ctx.normalization_type = normalization_type
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.opt_flow_loss = opt_flow_loss

        est_cards = to_variable(np.zeros(len(y)))
        for node,i in node_dict.items():
            if normalization_type == "mscn":
                est_cards[i] = (torch.exp((yhat[i] + min_val)*(max_val-min_val)))
            elif normalization_type == "pg_total_selectivity":
                est_cards[i] = yhat[i]*subsetg.nodes()[node]["cardinality"]["total"]
            else:
                assert False

        # TODO: simplify get_edge_costs further
        predC, dgdxT = get_edge_costs(subsetg, est_cards, node_dict, edge_dict,
                normalization_type, min_val, max_val)
        ctx.dgdxT = dgdxT

        # calculate flow loss
        G,Gv,Q = constructG(subsetg, predC, node_dict, edge_dict, final_node)

        mat_start = time.time()
        invG = torch.inverse(G)
        v = invG @ Gv
        ctx.save_for_backward(yhat, G, Gv, Q, trueC, invG, v)
        ctx.edge_dict = edge_dict
        ctx.node_dict = node_dict

        left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
        right = Q @ (v)
        loss = left @ trueC @ right

        if loss.item() == 0.0:
            print("flow loss was zero!")
            print(true_vals)
            print(preds)
            pdb.set_trace()
        loss = loss.reshape(1,1)

        # print("flow loss took: ", time.time()-start)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return gradients wrt preds, and bunch of Nones
        '''
        start = time.time()
        assert ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]

        # should we make sure these are contiguous in memory etc?
        yhat, G, Gv, Q, trueC, invG, v = ctx.saved_tensors
        subsetg = ctx.subsetg
        edge_dict = ctx.edge_dict
        node_dict = ctx.node_dict
        N = len(subsetg.nodes()) - 1
        M = len(subsetg.edges())

        dfdg = torch.zeros((M,M))

        # because this term is reused many times in the calculation below

        # compute each row
        rstart = time.time()
        QinvG = Q @ invG
        QinvG = QinvG.detach().numpy()
        v = v.detach().numpy()

        for edge, idx in edge_dict.items():
            # corresponding to ith cost edge
            row = compute_dfdg_row_np(idx, edge, node_dict, QinvG,
                    v, subsetg)
            dfdg[idx,:] = torch.from_numpy(row)

            # row = compute_dfdg_row(idx, edge, node_dict, QinvG,
                    # v, subsetg)
            # dfdg[idx,:] = row

            ## debug code
            # row_true = compute_dfdg_row_orig(idx, edge, node_dict, Q, G, invG, v, subsetg)
            # if np.linalg.norm(np.array(row - row_true)) > 1:
                # print(edge)
                # print(i)
                # # print(row)
                # # print(row_true)
                # print("norm diff: ", np.linalg.norm(np.array(row-row_true)))
            # else:
                # print(edge)

        # print("computing dfdg took: ", time.time()-rstart)

        dCdg = 2 * (dfdg @ (trueC @ (Q @ v)))

        # update_list("dCdg_grad.pkl", dCdg.detach().numpy())
        # print("backwards, dCdg: ", dCdg)

        yhat_grad = ctx.dgdxT @ dCdg
        print("flow loss backward took: ", time.time()-start)
        pdb.set_trace()
        yhat_grad /= ctx.opt_flow_loss
        return yhat_grad,None, None, None, None, None,None,None,None,None,None