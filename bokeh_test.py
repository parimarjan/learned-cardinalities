import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from scripts.parse_results import *
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
import networkx as nx
from cardinality_estimation.algs import *
import numpy as np
from cardinality_estimation.losses import *
#from cardinality_estimation.join_loss import JoinLoss, get_join_cost_sql, get_leading_hint
from cardinality_estimation.join_loss import *
from cardinality_estimation.nn import update_samples
import pdb

from sql_rep.utils import nodes_to_sql, path_to_join_order
#from cvxopt import matrix, solvers
#import cvxopt
import cvxpy as cp
import time
import copy
from multiprocessing import Pool
# FIXME: separate jupyter utils files especially for plotting utils

# import networkx as nx
from bokeh.io import output_file, show
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

def get_qerr(true, est):
    qerrs = []
    for k,y in true.items():
        yhat = est[k]
        qerrs.append(max( y / yhat, yhat / y))
    return sum(qerrs) / len(qerrs)

def eval_alg(sql, y, yhat, join_graph):
    env = JoinLoss(COST_MODEL, USER, "", "localhost", 5432, "imdb")

    print("qerr: ", get_qerr(y, yhat))
    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                env.compute_join_order_loss([sql], [join_graph],
                        [y], [yhat], None, True, num_processes=1, postgres=True, pool=None)

    print("jerr: {}".format(est_costs[0]-opt_costs[0]))
    plot_explain_join_order(est_plans[0], y, yhat, None, "Plan based on Estimates")
    plt.show()
    # FIXME:
    #plot_explain_join_order(opt_plans[0], y, y, None, "Plan based on true values")
    plot_explain_join_order(opt_plans[0], {}, {}, None, "Plan based on true values")

    plt.show()
    return opt_costs[0], opt_plans[0]

def get_all_errors(y, yhat, qrep):
    env = JoinLoss(COST_MODEL, USER, "", "localhost", 5432, "imdb")
    flow_env = PlanError(COST_MODEL, "flow-loss")
    plan_env = PlanError(COST_MODEL, "plan-loss", USER, "", "localhost", 5432, "imdb")
    qerr = get_qerr(y, yhat)
    sql = qrep["sql"]
    join_graph = qrep["join_graph"]
    opt_costs, est_costs, est_plans, opt_plans, est_sqls, opt_sqls = flow_env.compute_loss([qrep], [yhat])
    flow_err = est_costs[0] - opt_costs[0]
    opt_costs, est_costs, est_plans, opt_plans, est_sqls, opt_sqls = plan_env.compute_loss([qrep], [yhat],
                                                                true_cardinalities=[y], join_graphs=[join_graph],
                                                                                          pool=None)
    plan_err = est_costs[0] - opt_costs[0]
    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
            env.compute_join_order_loss([sql], [join_graph],
            [y], [yhat], None, True, num_processes=1, postgres=True, pool=None)
    join_err = est_costs[0] - opt_costs[0]

    return qerr,flow_err,plan_err,join_err

def plot_flow_path(edges, subsetg, solx, final_node, source_node, fn=None):
    source_node = tuple("s")
    cur_node = final_node
    edge_dict = {}
    edge_widths = {}
    for i, e in enumerate(edges):
        edge_dict[e] = i
        edge_widths[e] = solx[i]
        print(e, type(e))
        edge_dict[tuple((e[1], e[0]))] = i
        edge_widths[tuple((e[1], e[0]))] = solx[i]

    all_edges = []
    best_path_nodes = []
    while True:
        out_edges = subsetg.out_edges(cur_node)
        min_cost_edge = None
        min_cost = -100000000000

        for edge in out_edges:
            idx = edge_dict[edge]
            wt = solx[idx]
            if wt > min_cost:
                min_cost_edge = edge
                min_cost = wt
        all_edges.append(min_cost_edge)
        best_path_nodes.append(cur_node)
        cur_node = min_cost_edge[1]
        all_edges.append(min_cost_edge)
        if cur_node == source_node:
            best_path_nodes.append(cur_node)
            break

    print("going to draw flow path")
    draw_graph(subsetg, highlight_nodes=best_path_nodes, edge_widths=edge_widths, save_to=fn)

#QUERY_DIR = "./our_dataset/queries/"
#query = "6a/6a110.pkl"

#QUERY_DIR = "./so_workload/"
QUERY_DIR = "./our_dataset/queries/4a/"
query = "4a100.pkl"

#COST_MODEL = "nested_loop_index8"
COST_MODEL = "cm1"
COST_KEY = COST_MODEL + "cost"


#QUERY_DIR = "./debug_sqls/"
#query = "1.pkl"
#query = "2.pkl"

qfn = QUERY_DIR + query
postgres = Postgres()
true_alg = TrueCardinalities()

USER = "pari"
PWD = ""
HOST = "localhost"
PORT = 5432
DB_NAME = "imdb"
SAVE_DIR = "./QueryLabPlots/"

# for join loss computations
# pool = Pool(1)

qrep = load_sql_rep(qfn)
join_graph = qrep["join_graph"]
subset_graph = qrep["subset_graph"]
sql = qrep["sql"]
print(sql)
update_samples([qrep], 0, COST_MODEL, 1, DB_NAME)

subsetg = copy.deepcopy(qrep["subset_graph"])
#add_single_node_edges(subsetg)
qrep["subset_graph"] = subsetg
final_node = [n for n,d in subsetg.in_degree() if d==0][0]
source_node = tuple("s")

# fn = SAVE_DIR + "SubqueryGraphWithoutSource.png"
# draw_graph(subsetg, save_to=fn)
# G=nx.karate_club_graph()

# subsetg = nx.Graph(subsetg)
G = subsetg
mapping = {}
for node in G.nodes():
    mapping[node] = str(node)
G = nx.relabel_nodes(G, mapping)

pos = nx.nx_pydot.pydot_layout(G , prog='dot')
maxx = 0
minx = 10000000
maxy = 0
miny = 10000000
for k,v in pos.items():
    if v[0] > maxx:
        maxx = v[0]
    if v[0] < minx:
        minx = v[0]
    if v[1] > maxy:
        maxy = v[1]
    if v[1] < miny:
        miny = v[1]

for k,v in pos.items():
    # v[0] /= maxx
    # v[1] /= maxy
    # x = v[0] / maxx
    # y = v[1] / maxy
    x = (v[0] - minx) / (maxx - minx)
    y = (v[1] - miny) / (maxy - miny)
    pos[k] = tuple([x,y])

# pos = nx.spring_layout(G)

# G = qrep["join_graph"]
# print(type(G))
# print(G.nodes())
# pdb.set_trace()
# plot = Plot(plot_width=400, plot_height=400,
            # x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
# plot.title.text = "Graph Interaction Demonstration"

# graph_renderer = from_networkx(G, nx.spring_layout)
# graph_renderer = from_networkx(G, nx.nx_pydot.graphviz_layout)
# graph_renderer = from_networkx(G, pos)
# plot.renderers.append(graph_renderer)
# output_file("interactive_graphs.html")
# show(plot)

from bokeh.io import output_file, show
from bokeh.models import Ellipse, GraphRenderer, StaticLayoutProvider
from bokeh.palettes import Spectral8
from bokeh.plotting import figure

# N = 8
# node_indices = list(range(N))

# plot = figure(title='Graph Layout Demonstration', x_range=(-1.1,1.1),
        # y_range=(-1.1,1.1),
                      # tools='', toolbar_location=None)

plot = figure(title='Graph Layout Demonstration', x_range=(-0.1,1.1),
        y_range=(-0.1,1.1),
                      tools='', toolbar_location=None)

# plot = figure(title='Graph Layout Demonstration')
        # x_range=(-1.1,5.1),
        # y_range=(-1.1,5.1),
graph = GraphRenderer()

node_indices = list(G.nodes())

graph.node_renderer.data_source.add(node_indices, 'index')
graph.node_renderer.data_source.add(Spectral8, 'color')
graph.node_renderer.glyph = Ellipse(height=0.05, width=0.05,
       fill_color='color')

start = []
end = []
for e in G.edges():
    start.append(e[0])
    end.append(e[1])
graph.edge_renderer.data_source.data = dict(
           start=start,
               end=end)

### start of layout code
# circ = [i*2*math.pi/32 for i in range(len(node_indices))]
# x = [math.cos(i) for i in circ]
# y = [math.sin(i) for i in circ]
# graph_layout = dict(zip(node_indices, zip(x, y)))
# print(graph_layout)

graph_layout = pos
# print(pos)

graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

plot.renderers.append(graph)

output_file('graph.html')
show(plot)
