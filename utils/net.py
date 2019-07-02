import torch
from torch import nn
import torch.nn.functional as F

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, n_input, n_hidden, n_output):
        super(SimpleRegression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=True),
            nn.LeakyReLU()
        )

        # self.layer2 = nn.Sequential(
            # nn.Linear(n_hidden, n_output, bias=True),
            # nn.Sigmoid()
        # )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        return output
