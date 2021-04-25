import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scripts.parse_results import *
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from cardinality_estimation.algs import *
import numpy as np
from cardinality_estimation.losses import *
#from cardinality_estimation.join_loss import JoinLoss, get_join_cost_sql,
#get_leading_hint
from cardinality_estimation.join_loss import *
from cardinality_estimation.nn import update_samples

from sql_rep.utils import nodes_to_sql, path_to_join_order
#from cvxopt import matrix, solvers
#import cvxopt
import cvxpy as cp
import time
import copy
from multiprocessing import Pool

def get_flows(subsetg, cost_key):

    # FIXME: assuming subsetg  is S->D; construct_lp expects it to be D->S.
    edges, costs, A, b, G, h = construct_lp(subsetg, cost_key=cost_key)
    n = len(edges)
    P = np.zeros((len(edges),len(edges)))
    for i,c in enumerate(costs):
        P[i,i] = c

    q = np.zeros(len(edges))
    x = cp.Variable(n)
    #prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
    #                 [G @ x <= h,
    #                  A @ x == b])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [A @ x == b])
    prob.solve(verbose=False)
    flows = np.array(x.value)
    return flows, edges

def draw_plan_graph(subsetg, y, cost_model, ax=None,
        source_node=SOURCE_NODE, final_node=None, font_size=40,
        cbar_fontsize=24, cax=None, fig=None, width=None, edge_color=None,
        bold_opt_path=True, bold_path=None):

    cost_key = "tmp_cost"
    subsetg = subsetg.reverse()
    pg_total_cost = compute_costs(subsetg, cost_model,
                                cost_key=cost_key,
                                ests=y)

    flows, edges = get_flows(subsetg, cost_model+cost_key)
    # reverse back
    subsetg = subsetg.reverse()

    edge_colors = []
    for edge in subsetg.edges(data=True):
        edge_colors.append(edge[2][cost_model+cost_key])

    vmin = min(edge_colors)
    vmax = max(edge_colors)

    assert len(edge_colors) == len(flows)

    # MIN: 2...6
    MIN_WIDTH = 1.0
    MAX_WIDTH = 30.0
    NEW_RANGE = MAX_WIDTH - MIN_WIDTH
    OLD_RANGE = max(flows) - min(flows)

    edge_widths = {}
    for i, x in enumerate(flows):
        #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        normx = (((x - min(flows))*NEW_RANGE) / OLD_RANGE) + MIN_WIDTH
        #edge_widths.append(normx)
        edge_widths[edges[i]] = normx

    widths = []
    for edge in subsetg.edges():
        key = tuple([edge[1], edge[0]])
        widths.append(edge_widths[key])

    opt_labels_list = nx.shortest_path(subsetg, source_node,
            final_node, weight=cost_model+cost_key)
    opt_labels = {}
    for n in subsetg.nodes(data=True):
        if n[0] in opt_labels_list:
            opt_labels[n[0]] = n[1]["label"]

    pos = nx.nx_pydot.pydot_layout(subsetg, prog="dot")
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

    if ax is None:
        # plt.style.use("ggplot")
        fig, ax = plt.subplots(1,1,figsize=(30,20))

    labels = nx.get_node_attributes(subsetg, 'label')

    nx.draw_networkx_labels(subsetg, pos=pos,
            labels=labels,
            ax=ax, font_size=font_size,
            bbox=dict(facecolor="w", edgecolor='k', boxstyle='round,pad=0.1'))

    if bold_opt_path:
        nx.draw_networkx_labels(subsetg, pos=pos,
                labels=opt_labels,
                ax=ax,
                font_size=font_size, bbox=dict(facecolor="w", edgecolor='k',
                    lw=font_size/2,
                    boxstyle='round,pad=0.5', fill=True))
    if bold_path:
        bold_labels = {}
        for n in subsetg.nodes(data=True):
            if n[0] in bold_path:
                bold_labels[n[0]] = n[1]["label"]
        nx.draw_networkx_labels(subsetg, pos=pos,
                labels=bold_labels,
                ax=ax,
                font_size=font_size, bbox=dict(facecolor="w", edgecolor='k',
                    lw=font_size/2,
                    boxstyle='round,pad=0.5', fill=True))

    if width is not None:
        widths = width

    if edge_color is not None:
        edge_colors = edge_color

    edges = nx.draw_networkx_edges(subsetg, pos, edge_color=edge_colors,
            width=widths, ax = ax, edge_cmap=cm,
            arrows=True,
            arrowsize=font_size / 2,
            arrowstyle='simple',
            min_target_margin=5.0)

    if edge_color is None:
        plt.style.use("seaborn-white")
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        if fig is None:
            cbar = plt.colorbar(sm, aspect=50, orientation="horizontal", pad =
                    0.02)
        else:
            cbar = fig.colorbar(sm, ax=ax,
                    pad = 0.02,
                    aspect=50,
                    orientation="horizontal")

        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label("Cost", fontsize=font_size)
        cbar.ax.xaxis.get_offset_text().set_fontsize(font_size)
    plt.tight_layout()

def simple_plot_explain(explain, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,6))

    G = explain_to_nx(explain)
    pos = graphviz_layout(G, prog='dot')
    G = G.reverse()

    nx.draw(G, pos,
               node_size= 1800,
               node_color = "w",
               alpha=0.5,
               ax=ax)

    labels = {}
    for k, v in pos.items():
        try:
            int(k)
            labels[k] = "$\Join$"
        except:
            labels[k] = k

    nx.draw_networkx_labels(G, pos, labels,
            font_size=20,
            font_color="k",
            ax = ax)

    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
