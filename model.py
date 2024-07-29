import torch
import torch.nn as nn

from sub_module.cde_function import CDEFunc
from sub_module.factor_encoder import FactorEncoder
from sub_module.neural_cde import NeuralCDE


class NCDENow(nn.Module):

    def __init__(self, fe_type, cde_type, input_size, hidden_size, hidden_hidden_size, output_size, n_factors, n_layers,
                 ode_method, device):
        super(NCDENow, self).__init__()
        self.device = device
        self.factor_encoder = FactorEncoder(fe_type, input_size, hidden_size, n_factors, n_layers, device)
        self.cde_func = CDEFunc(cde_type, input_size, hidden_size, hidden_hidden_size, n_layers)
        self.neural_cde = NeuralCDE(self.cde_func, input_size, hidden_size, output_size, n_factors, ode_method)

    def forward(self, x, coeffs, factors):
        z = self.factor_encoder(x, factors)
        pred_alpha, pred_beta = self.neural_cde(coeffs)
        pred_y = pred_alpha + torch.sum(pred_beta * z, dim=1).unsqueeze(1)

        return pred_y, pred_alpha, pred_beta
