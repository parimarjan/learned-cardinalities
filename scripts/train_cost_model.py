import sys
sys.path.append(".")
import pickle
import glob
import argparse
import pandas as pd
import os
from utils.utils import *
import pdb
from matplotlib import gridspec
from matplotlib import pyplot as plt
from db_utils.utils import *
from db_utils.query_storage import *
from parsing_utils import *
from cardinality_estimation.cost_dataset import CostDataset
from torch.utils import data
from utils.net import *
from sklearn.model_selection import train_test_split

from metric_learn import LMNN, NCA

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_file", type=str, required=False,
            default="./join_loss_data/1a.pkl")
    parser.add_argument("--query_dir", type=str, required=False,
            default="./our_dataset/queries")
    parser.add_argument("--feat_type", type=str, required=False,
            default="fcnn")
    parser.add_argument("--tfboard_dir", type=str, required=False,
            default=None)
    parser.add_argument("--lr", type=float, required=False,
            default=0.0001)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.2)
    parser.add_argument("--hidden_layer_multiple", type=float, required=False,
            default=2)
    parser.add_argument("--num_hidden_layers", type=int, required=False,
            default=1)
    parser.add_argument("--add_true", type=int, required=False,
            default=0)
    parser.add_argument("--mb_size", type=int, required=False,
            default=32)
    parser.add_argument("--max_epochs", type=int, required=False,
            default=10)
    parser.add_argument("--input_feat_type", type=int, required=False,
            default=1)
    parser.add_argument("--input_norm_type", type=int, required=False,
            default=1)
    parser.add_argument("--test_while_training", type=int, required=False,
            default=0)
    parser.add_argument("--learn_type", type=str, required=False,
            default="metric")

    return parser.parse_args()

def periodic_eval(net, loader, loss_func):

    losses = []
    for xbatch,ybatch in loader:
        preds = net(xbatch).squeeze(1)
        loss = loss_func(preds, ybatch).cpu().detach().numpy()
        losses.append(loss)
    return sum(losses) / len(losses)

def main():
    mapping = qkey_map(args.query_dir)
    assert len(mapping) != 0
    training_data = load_object(args.training_data_file)
    print(len(training_data["jloss"]), len(np.unique(training_data["jloss"])))
    print("num queries: ", len(set(training_data["key"])))
    tr_keys, test_keys, tr_ests, test_ests, tr_costs, test_costs, tr_ratios, \
            test_ratios = \
            train_test_split(training_data["key"], training_data["est"],
                    training_data["jloss"], training_data["jratio"], random_state=1234,
                    test_size=args.test_size)
    # split it
    print(len(tr_keys), len(test_keys))
    train_dataset = CostDataset(mapping, tr_keys, tr_ests, tr_costs, tr_ratios,
            args.feat_type, input_feat_type = args.input_feat_type,
            add_true=args.add_true)
    test_dataset = CostDataset(mapping, test_keys, test_ests, test_costs,
            test_ratios,
            args.feat_type, input_feat_type = args.input_feat_type,
            add_true=args.add_true)
    train_loader = data.DataLoader(train_dataset,
            batch_size=args.mb_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset,
            batch_size=10000, shuffle=False, num_workers=0)
    print("num train: {}, num test: {}".format(len(train_dataset),
        len(test_dataset)))

    # TODO: make this the test set
    eval_loader = data.DataLoader(train_dataset,
            batch_size=10000, shuffle=False, num_workers=0)

    inp_len = len(train_dataset[0][0])
    print("inp len: ", inp_len)

    net = CostModelNet(inp_len, args.hidden_layer_multiple, 1,
            num_hidden_layers=args.num_hidden_layers)
    loss_func = torch.nn.MSELoss()

    if args.tfboard_dir:
        make_dir(tfboard_dir)
        tfboard = TensorboardSummaries(tfboard_dir + "/tf_cost_logs/" +
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        tfboard.init()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(0, args.max_epochs):

        if args.test_while_training or \
                epoch == args.max_epochs-1:
                    print("epoch: {}, train loss: {}, test_loss: {}".format(epoch, periodic_eval(net,
                        eval_loader, loss_func), periodic_eval(net, test_loader,
                            loss_func)))
        else:
            print("epoch: {}, N: {} train loss: {}".format(epoch,
                len(train_dataset), periodic_eval(net,
                    eval_loader, loss_func)))

                # train loop
        for _, (xbatch, ybatch) in enumerate(train_loader):
            pred = net(xbatch).squeeze(1)
            assert pred.shape == ybatch.shape
            loss = loss_func(pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            # if args.clip_gradient is not None:
                # clip_grad_norm_(self.net.parameters(), args.clip_gradient)
            optimizer.step()

        torch.save(net, "./cm_fcnn.pt")

if __name__ == "__main__":
    args = read_flags()
    main()
