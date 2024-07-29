import torch.nn as nn

from encoder.enc_factor_analysis import FactorAnalysisEncoder
from encoder.enc_gru import GRUEncoder


class FactorEncoder(nn.Module):

    def __init__(self, fe_type, input_size, hidden_size, n_factors, n_layers, device):
        super(FactorEncoder, self).__init__()
        self.model = globals()[fe_type]

        # GRU encoder
        if self.model == GRUEncoder:
            self.model = self.model(input_size, hidden_size, n_factors, n_layers, device)

        # Factor Analysis
        elif self.model == FactorAnalysisEncoder:
            self.model = self.model()

    def forward(self, x, factors):
        return self.model(x, factors)
