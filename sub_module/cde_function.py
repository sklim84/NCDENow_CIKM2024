import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

import torch.nn as nn

from cde_func.cde_mlp import MLPCDEFunc
from cde_func.cde_gru import GRUCDEFunc


class CDEFunc(nn.Module):

    def __init__(self, cde_type, input_size, hidden_size, hidden_hidden_size, n_layers):
        super(CDEFunc, self).__init__()
        self.model = globals()[cde_type]

        # MLP based CDE function
        if self.model == MLPCDEFunc:
            self.model = self.model(input_size, hidden_size, hidden_hidden_size, n_layers)

        # GRU based CDE function
        elif self.model == GRUCDEFunc:
            self.model = self.model(input_size, hidden_size, hidden_hidden_size, n_layers)

    def forward(self, t, gamma):
        return self.model(t, gamma)
