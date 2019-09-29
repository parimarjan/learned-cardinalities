import torch
from torch import nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FatHydra(torch.nn.Module):
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_tables):
        super(FatHydra, self).__init__()
        # linear layer + sigmoid for each num_tables option
        self.pred_layers = []
        print("creating FatHydra for {} tables".format(num_tables))

        n_hidden = int(input_width * hidden_width_multiple)
        self.layer1 = nn.Sequential(
            nn.Linear(input_width, n_hidden, bias=True),
            nn.LeakyReLU()
        ).to(device)

        for i in range(0, num_tables):
            final_layer = nn.Sequential(
                nn.Linear(n_hidden, n_output, bias=True),
                nn.Sigmoid()
            ).to(device)
            self.pred_layers.append(final_layer)

        self.pred_layers = nn.ModuleList(self.pred_layers)

        assert num_tables == len(self.pred_layers)

    def forward(self, x_all):
        outputs = []
        x_all = self.layer1(x_all)
        for x in x_all:
            num_tables = int(x[-1].item())
            output = self.pred_layers[num_tables-1](x)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs

class Hydra(torch.nn.Module):
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_tables, linear):
        super(Hydra, self).__init__()
        self.layers = []
        # linear layer + sigmoid for each num_tables option
        self.pred_layers = []
        print("creating Hydra for {} tables".format(num_tables))

        n_hidden = int(input_width * hidden_width_multiple)
        if linear:
            layer1 = nn.Sequential(
                nn.Linear(input_width, n_hidden, bias=True),
            ).to(device)
        else:
            layer1 = nn.Sequential(
                nn.Linear(input_width, n_hidden, bias=True),
                nn.LeakyReLU()
            ).to(device)
        self.layers.append(layer1)

        for i in range(num_tables-1):
            print(i)
            if linear:
                layer = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden, bias=True),
                ).to(device)
            else:
                layer = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden, bias=True),
                    nn.LeakyReLU()
                ).to(device)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

        for i in range(0, num_tables):
            final_layer = nn.Sequential(
                nn.Linear(n_hidden, n_output, bias=True),
                nn.Sigmoid()
            ).to(device)
            self.pred_layers.append(final_layer)

        self.pred_layers = nn.ModuleList(self.pred_layers)

        assert len(self.layers) == len(self.pred_layers)

    def forward(self, x_all):
        outputs = []
        for x in x_all:
            num_tables = int(x[-1].item())
            # print("num tables: ", num_tables)
            output = x
            for i, layer in enumerate(self.layers):
                if i >= num_tables:
                    break
                output = layer(output)
            output = self.pred_layers[num_tables-1](output)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs

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
            n_output, num_hidden_layers=1):
        super(SimpleRegression, self).__init__()
        n_hidden = int(input_width * hidden_width_multiple)
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
