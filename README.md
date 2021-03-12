# Testbed for Cardinality Estimation

## Steps for running a simple example with flow-loss

### Download the Cardinality Estimation Benchmark

Follow instructions at https://github.com/Cardinality-Estimation-Benchmark/ceb

This is neccessary as constructing the graph for Flow-Loss relies on the way we
store queries in CEB etc.

### Compile the Flow-Loss implementation

```bash
cd flow_loss_cpp
make
```


### Simple execution example

After the CEB has been set up, and the queries extracted to query/imdb, run the
following command to train a simple neural for CE with Flow-Loss. Use the
appropriate PostgreSQL user / password (PostgreSQL should not be neccessary in
    general, it is just used for featurizing the queries.)

```bash
python3 simple_flow_loss.py --query_dir queries/imdb/ --user pari --pwd password --query_template 1a -n 100 --normalize_flow_loss 1
```
