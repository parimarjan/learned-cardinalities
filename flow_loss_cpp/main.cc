#include <iostream>
#include <cmath>
#include <math.h>
#include "omp.h"

#define NILJ_CONSTANT 0.001
#define SOURCE_NODE_CONST 100000

void get_dfdg_par(int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *QinvG, float *v, float *dfdg, int batch,
    int batch_size)
{
  //printf("thread %d\n", omp_get_thread_num());

  int hnode, tnode, start, end;
  float vh, vt, qgh, qgt;
  start = batch*batch_size;
  end = start+batch_size;
  if (end > num_edges) end = num_edges;

  for (int i = start; i < end; i++) {
    hnode = edges_head[i];
    tnode = edges_tail[i];
    vh = v[hnode];
    if (tnode != SOURCE_NODE_CONST) {
      vt = v[tnode];
    } else vt = 0.0;
    //printf("edge: %d, hnode: %d, tnode: %d, vh: %f, vt: %f\n", i, hnode, tnode, vh, vt);

    for (int j = 0; j < num_edges; j++) {
      if (i == j) {
        qgh = 1.0;
        qgt = -1.0;
      } else {
        qgh = 0.0;
        qgt = 0.0;
      }

      dfdg[i*num_edges + j] = vh*(qgh - QinvG[j*num_nodes + hnode]);
      if (tnode != SOURCE_NODE_CONST) {
        dfdg[i*num_edges + j] += vh*(QinvG[j*num_nodes + tnode]);
        dfdg[i*num_edges + j] += vt*(qgt + QinvG[j*num_nodes + hnode] \
            - QinvG[j*num_nodes+tnode]);
      }
    }
  }
}

extern "C" void get_dfdg (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *QinvG, float *v, float *dfdg,
    int num_workers)
{
  //printf("get dfdg, %d, %d\n", num_edges, num_nodes);
  //int hnode, tnode;
  //float vh, vt, qgh, qgt;
  int batch_size, num_batches;
  batch_size = num_edges / num_workers;

  int rem = num_edges % batch_size;
  num_batches = ceil(num_edges / batch_size);
  //printf("rem: %d\n", rem);
  if (rem != 0) num_batches += 1;

  #pragma omp parallel for num_threads(num_workers)
  for (int batch = 0; batch < num_batches; batch++) {
    get_dfdg_par(num_edges, num_nodes, edges_head,
        edges_tail, QinvG, v, dfdg, batch, batch_size);
  }
}

/* fills in tqv, Mx1 array
 * */
extern "C" void get_tqv (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *v, float *trueC, int const_mul,
    float *tqv)
{
  int head_node, tail_node;

  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    /* will set tqv[i] now */
    tqv[i] = Q[i*num_nodes + head_node] * v[head_node];
    //printf("head node: %d, tail node: %d\n", head_node, tail_node);
    if (tail_node != SOURCE_NODE_CONST) {
      tqv[i] += Q[i*num_nodes + tail_node] * v[tail_node];
    }
    tqv[i] *= trueC[i]*const_mul;
  }
}

/* fills in tqv, len 1 array
 * */
extern "C" void get_qvtqv (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *v, float *trueC,
    float *tqv)
{
  int head_node, tail_node;
  float cur_val;

  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    /* will set tqv[i] now */
    cur_val = 0.0;
    cur_val += Q[i*num_nodes + head_node] * v[head_node];
    if (tail_node != SOURCE_NODE_CONST) {
      cur_val += Q[i*num_nodes + tail_node] * v[tail_node];
    }
    cur_val = cur_val * cur_val;
    //printf("%d, %d\n", head_node, tail_node);
    tqv[0] += trueC[i]*cur_val;
  }
}

/* fills in QinvG mat mul, knowing the sparsity pattern of Q
 * M: num_edges, N: num_nodes
 * Q: M x N: each row has zeros except in the nodes making that edge
 * invG: NxN not sparse
 * QinvG: MxN
 * */
extern "C" void get_qinvg (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *invG,
    float *QinvG)
{
  int head_node, tail_node, idx;
  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    for (int j = 0; j < num_nodes; j++)
    {
      idx = i*num_nodes + j;
      QinvG[idx] = Q[i*num_nodes + head_node]*invG[head_node*num_nodes + j];
      if (tail_node != SOURCE_NODE_CONST) {
        QinvG[idx] += Q[i*num_nodes + tail_node]*invG[tail_node*num_nodes + j];
      }
    }
  }
}


void get_opt_var_par(
    float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT,
    float *G, float *Q, int batch, int batch_size)
{
  double card1, card2, hash_join_cost, nilj_cost;
  int node1, node2, head_node, tail_node, start, end;
  start = batch*batch_size;
  end = start+batch_size;
  if (end > num_edges) end = num_edges;
  printf("start = %d, end = %d\n", start, end);

  for (int i = start; i < end; i++) {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    if (edges_cost_node1[i] == SOURCE_NODE_CONST) {
      costs[i] = 1.0;
      // still need to construct the matrices stuff
      // tail node was the final node, which we don't have in Q / G
      Q[i*num_nodes + head_node] = 1 / costs[i];
      G[head_node*num_nodes + head_node] += 1 / costs[i];
      continue;
    }
    node1 = edges_cost_node1[i];
    node2 = edges_cost_node2[i];
    card1 = ests[node1] + 1.0;
    card2 = ests[node2] + 1.0;
    hash_join_cost = card1 + card2;
    if (nilj[i] == 1) {
      nilj_cost = card2 + NILJ_CONSTANT*card1;
    } else if (nilj[i] == 2) {
      nilj_cost = card1 + NILJ_CONSTANT*card2;
    } else {
      nilj_cost = 10000000000.0;
    }
    if (hash_join_cost < nilj_cost) costs[i] = hash_join_cost;
    else costs[i] = nilj_cost;
    float cost = costs[i];

    /* time to update the derivatives */
    //if (normalization_type == 0) continue;

    /* time to compute gradients */
    if (normalization_type == 1) {
      float total1 = totals[node1];
      float total2 = totals[node2];
      if (hash_join_cost < nilj_cost) {
        //- (a / (ax_1 + bx_2)**2)
        dgdxT[node1*num_edges + i] = - (total1 / (hash_join_cost*hash_join_cost));
        dgdxT[node2*num_edges + i] = - (total2 / (hash_join_cost*hash_join_cost));
      } else {
          if (nilj[i] == 1) {
            float num1 = total1*NILJ_CONSTANT;
            dgdxT[node1*num_edges + i] = - (num1 / (costs[i]*costs[i]));
            dgdxT[node2*num_edges + i] = - (total2 / (costs[i]*costs[i]));
          } else {
            //# node 2
            //# - (a / (ax_1 + bx_2)**2)
            float num2 = total2*NILJ_CONSTANT;
            dgdxT[node1*num_edges + i] = - (total1 / (costs[i]*costs[i]));
            //# - (b / (ax_1 + bx_2)**2)
            dgdxT[node2*num_edges + i] = - (num2 / (costs[i]*costs[i]));
          }
      }
    } else {
      // log normalization type
      if (hash_join_cost <= nilj_cost) {
          //- (ae^{ax} / (e^{ax} + e^{ax2})**2)
           //e^{ax} is just card1
          dgdxT[node1*num_edges + i] = - (max_val*card1) / (hash_join_cost*hash_join_cost);
          dgdxT[node2*num_edges + i] = - (max_val*card2) / (hash_join_cost*hash_join_cost);
      } else {
          // index nested loop join
          if (nilj[i]  == 1) {
              dgdxT[node1*num_edges + i] = - (max_val*card1*NILJ_CONSTANT) / (cost*cost);
              dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          } else {
              //float num2 = card2*NILJ_CONSTANT;
              dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
              dgdxT[node2*num_edges + i] = - (max_val*card2*NILJ_CONSTANT) / (cost*cost);
          }
      }
    }

    cost = 1.0 / costs[i];

    /* construct G, Q */
    Q[i*num_nodes + head_node] = cost;
    G[head_node*num_nodes + head_node] += cost;

    if (tail_node != SOURCE_NODE_CONST) {
        Q[i*num_nodes + tail_node] = -cost;
        G[tail_node*num_nodes + tail_node] += cost;
        G[head_node*num_nodes + tail_node] -= cost;
        G[tail_node*num_nodes + head_node] -= cost;
    }
  }
}

/* Note: the values to be returned are passed in as well.
 */
extern "C" void get_optimization_variables(
    float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT,
    float *G, float *Q, int num_workers)
{
  //int num_workers = 1;
  int batch_size, num_batches;
  batch_size = num_edges / num_workers;

  int rem = num_edges % batch_size;
  num_batches = ceil(num_edges / batch_size);
  if (rem != 0) num_batches += 1;

  #pragma omp parallel for num_threads(num_workers)
  for (int batch = 0; batch < num_batches; batch++) {
    //get_dfdg_par(num_edges, num_nodes, edges_head,
        //edges_tail, QinvG, v, dfdg, batch, batch_size);
    get_opt_var_par(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, edges_head, edges_tail,
        nilj, num_nodes, num_edges, costs, dgdxT, G, Q, batch, batch_size);
  }
}

extern "C" void get_opt_debug(
    float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT,
    float *G, float *Q)
{
  double card1, card2, hash_join_cost, nilj_cost;
  int node1, node2, head_node, tail_node;

  for (int i = 0; i < num_edges; i++) {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    if (edges_cost_node1[i] == SOURCE_NODE_CONST) {
      costs[i] = 1.0;
      // still need to construct the matrices stuff
      // tail node was the final node, which we don't have in Q / G
      Q[i*num_nodes + head_node] = 1 / costs[i];
      G[head_node*num_nodes + head_node] += 1 / costs[i];
      continue;
    }
    node1 = edges_cost_node1[i];
    node2 = edges_cost_node2[i];
    card1 = ests[node1] + 1.0;
    card2 = ests[node2] + 1.0;
    hash_join_cost = card1 + card2;
    if (nilj[i] == 1) {
      nilj_cost = card2 + NILJ_CONSTANT*card1;
    } else if (nilj[i] == 2) {
      nilj_cost = card1 + NILJ_CONSTANT*card2;
    } else {
      nilj_cost = 10000000000.0;
    }
    if (hash_join_cost < nilj_cost) costs[i] = hash_join_cost;
    else costs[i] = nilj_cost;
    float cost = costs[i];

    /* time to update the derivatives */
    //if (normalization_type == 0) continue;

    /* time to compute gradients */
    if (normalization_type == 1) {
      float total1 = totals[node1];
      float total2 = totals[node2];
      if (hash_join_cost < nilj_cost) {
        //- (a / (ax_1 + bx_2)**2)
        dgdxT[node1*num_edges + i] = - (total1 / (hash_join_cost*hash_join_cost));
        dgdxT[node2*num_edges + i] = - (total2 / (hash_join_cost*hash_join_cost));
      } else {
          if (nilj[i] == 1) {
            float num1 = total1*NILJ_CONSTANT;
            dgdxT[node1*num_edges + i] = - (num1 / (costs[i]*costs[i]));
            dgdxT[node2*num_edges + i] = - (total2 / (costs[i]*costs[i]));
          } else {
            //# node 2
            //# - (a / (ax_1 + bx_2)**2)
            float num2 = total2*NILJ_CONSTANT;
            dgdxT[node1*num_edges + i] = - (total1 / (costs[i]*costs[i]));
            //# - (b / (ax_1 + bx_2)**2)
            dgdxT[node2*num_edges + i] = - (num2 / (costs[i]*costs[i]));
          }
      }
    } else {
      // log normalization type
      if (hash_join_cost <= nilj_cost) {
          //- (ae^{ax} / (e^{ax} + e^{ax2})**2)
           //e^{ax} is just card1
          dgdxT[node1*num_edges + i] = - (max_val*card1) / (hash_join_cost*hash_join_cost);
          dgdxT[node2*num_edges + i] = - (max_val*card2) / (hash_join_cost*hash_join_cost);
      } else {
          // index nested loop join
          if (nilj[i]  == 1) {
              dgdxT[node1*num_edges + i] = - (max_val*card1*NILJ_CONSTANT) / (cost*cost);
              dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          } else {
              //float num2 = card2*NILJ_CONSTANT;
              dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
              dgdxT[node2*num_edges + i] = - (max_val*card2*NILJ_CONSTANT) / (cost*cost);
          }
      }
    }

    cost = 1.0 / costs[i];

    /* construct G, Q */
    Q[i*num_nodes + head_node] = cost;
    G[head_node*num_nodes + head_node] += cost;

    if (tail_node != SOURCE_NODE_CONST) {
        Q[i*num_nodes + tail_node] = -cost;
        G[tail_node*num_nodes + tail_node] += cost;
        G[head_node*num_nodes + tail_node] -= cost;
        G[tail_node*num_nodes + head_node] -= cost;
    }
  }
}

