# Testbed for Cardinality Estimation

## Setup

A docker image to set up postgres. We can update the init script in ./docker to
set up the appropriate DB, and supply the relevant authentication args to
main.py. For now, I just create a dummy DB and populate it with synthetic data.

```bash
$ cd docker
$ sudo docker build -t card-est
$ sudo docker run --name card-db -p 5401:5432 -d card-est
```

Now, you can connect to the db with:

```bash
$ psql -U card_est -h localhost -p 5401 -d card_est
```

## Steps

  * Create new DB + generate synthetic data OR set up a pre-existing DB
  * Generate `interesting` cardinality queries (see cardinality_estimation/db_utils/Query)
  using templates defined in ./test_templates. These queries can be over a
  single table or multiple tables.
  * Maybe, generate all possible SubQueries from each Query (only relevant for
      multi-table joins)
  * Implement an algorithm as a subclass of
  cardinality_estimation/algs/CardinalityEstimationAlg. The train method can
  use a bunch of the CardinalitySamples in the training set (`query-feedback` based systems, e.g.,
      quicksel, neural nets etc.) or ignore those, and only use the
  data in the underlying table (wavelet based classifier, pgm etc.) or combine
  both of these.
  * Compare performance on the test set etc.

