import torch
from torch import nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LinearRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, input_width,
            n_output):
        super(LinearRegression, self).__init__()

        self.final_layer = nn.Sequential(
            nn.Linear(input_width, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        output = x
        output = self.final_layer(output)
        return output

class CostModelNet(torch.nn.Module):
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_hidden_layers=1, hidden_layer_size=None):
        super(CostModelNet, self).__init__()
        if hidden_layer_size is None:
            n_hidden = int(input_width * hidden_width_multiple)
        else:
            n_hidden = hidden_layer_size

        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Linear(input_width, n_hidden, bias=True),
            nn.LeakyReLU()
        ).to(device)
        self.layers.append(self.layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.LeakyReLU()
            ).to(device)
            self.layers.append(layer)

        self.final_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            # nn.Sigmoid()
        ).to(device)
        self.layers.append(self.final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_hidden_layers=1, hidden_layer_size=None):
        super(SimpleRegression, self).__init__()
        if hidden_layer_size is None:
            n_hidden = int(input_width * hidden_width_multiple)
        else:
            n_hidden = hidden_layer_size

        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Linear(input_width, n_hidden, bias=True),
            nn.LeakyReLU()
        ).to(device)
        self.layers.append(self.layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.LeakyReLU()
            ).to(device)
            self.layers.append(layer)

        self.final_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.layers.append(self.final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

# MSCN model, kipf et al.
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units,
            dropout=0.0):
        super(SetConv, self).__init__()
        self.dropout = dropout
        # doesn't really make sense to have this be bigger...
        # sample_hid = min(hid_units, 128)
        sample_hid = hid_units
        self.sample_mlp1 = nn.Sequential(
            nn.Linear(sample_feats, sample_hid),
            nn.ReLU()
        ).to(device)

        self.sample_mlp2 = nn.Sequential(
            nn.Linear(sample_hid, sample_hid),
            nn.ReLU()
        ).to(device)

        self.predicate_mlp1 = nn.Sequential(
            nn.Linear(predicate_feats, hid_units),
            nn.ReLU()
        ).to(device)

        self.predicate_mlp2 = nn.Sequential(
            nn.Linear(hid_units, hid_units),
            nn.ReLU()
        ).to(device)

        # join_hid = min(hid_units, 128)
        join_hid = hid_units
        self.join_mlp1 = nn.Sequential(
            nn.Linear(join_feats, join_hid),
            nn.ReLU()
        ).to(device)

        self.join_mlp2 = nn.Sequential(
            nn.Linear(join_hid, join_hid),
            nn.ReLU()
        ).to(device)

        total_hid = sample_hid + join_hid + hid_units

        self.out_mlp1 = nn.Sequential(
                nn.Linear(total_hid, hid_units),
                nn.ReLU()
        ).to(device)

        self.out_mlp2 = nn.Sequential(
                nn.Linear(hid_units, 1),
                nn.Sigmoid()
        ).to(device)

        self.drop_layer = nn.Dropout(self.dropout)

    def forward(self, samples, predicates, joins):

        hid_sample = self.sample_mlp1(samples)
        # hid_sample = self.drop_layer(hid_sample)
        hid_sample = self.sample_mlp2(hid_sample)

        hid_predicate = self.predicate_mlp1(predicates)
        hid_predicate = self.drop_layer(hid_predicate)
        hid_predicate = self.predicate_mlp2(hid_predicate)

        hid_join = self.join_mlp1(joins)
        # hid_join = self.drop_layer(hid_join)
        hid_join = self.join_mlp2(hid_join)

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)

        hid = self.out_mlp1(hid)
        # hid = self.drop_layer(hid)
        out = self.out_mlp2(hid)

        return out
