import torch
from torch import nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class LinearRegression(torch.nn.Module):
    # def __init__(self, input_width,
            # n_output):
        # super(LinearRegression, self).__init__()

        # self.final_layer = nn.Sequential(
            # nn.Linear(input_width, n_output, bias=True),
            # nn.Sigmoid()
        # ).to(device)

    # def forward(self, x):
        # output = x
        # output = self.final_layer(output)
        # return output

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
