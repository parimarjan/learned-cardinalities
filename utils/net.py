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
        # output = self.layer1(x)
        # output = self.layer2(output)
        # output = self.layer3(output)
        return output

# MSCN model, kipf et al.
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins):
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        # hid_sample = hid_sample * sample_mask  # Mask
        # hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        # sample_norm = sample_mask.sum(1, keepdim=False)
        # hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        # hid_predicate = hid_predicate * predicate_mask
        # hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        # predicate_norm = predicate_mask.sum(1, keepdim=False)
        # hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        # hid_join = hid_join * join_mask
        # hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        # join_norm = join_mask.sum(1, keepdim=False)
        # hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

    # def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # # samples has shape [batch_size x num_joins+1 x sample_feats]
        # # predicates has shape [batch_size x num_predicates x predicate_feats]
        # # joins has shape [batch_size x num_joins x join_feats]

        # hid_sample = F.relu(self.sample_mlp1(samples))
        # hid_sample = F.relu(self.sample_mlp2(hid_sample))
        # hid_sample = hid_sample * sample_mask  # Mask
        # hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        # sample_norm = sample_mask.sum(1, keepdim=False)
        # hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        # hid_predicate = F.relu(self.predicate_mlp1(predicates))
        # hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        # hid_predicate = hid_predicate * predicate_mask
        # hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        # predicate_norm = predicate_mask.sum(1, keepdim=False)
        # hid_predicate = hid_predicate / predicate_norm

        # hid_join = F.relu(self.join_mlp1(joins))
        # hid_join = F.relu(self.join_mlp2(hid_join))
        # hid_join = hid_join * join_mask
        # hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        # join_norm = join_mask.sum(1, keepdim=False)
        # hid_join = hid_join / join_norm

        # hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        # hid = F.relu(self.out_mlp1(hid))
        # out = torch.sigmoid(self.out_mlp2(hid))
        # return out
