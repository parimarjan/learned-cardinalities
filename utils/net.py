import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, n_input, n_hidden, n_output, num_hidden_layers=1):
        super(SimpleRegression, self).__init__()
        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=True),
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

        # self.layer2 = nn.Sequential(
            # nn.Linear(n_hidden, n_hidden, bias=True),
            # nn.LeakyReLU()
        # ).to(device)

        # self.layer3 = nn.Sequential(
            # nn.Linear(n_hidden, n_output, bias=True),
            # nn.Sigmoid()
        # ).to(device)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        # output = self.layer1(x)
        # output = self.layer2(output)
        # output = self.layer3(output)
        return output
