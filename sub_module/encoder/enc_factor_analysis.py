import torch.nn as nn


class FactorAnalysisEncoder(nn.Module):

    def __init__(self):
        super(FactorAnalysisEncoder, self).__init__()

    def forward(self, x, factors):
        # factors : pre-processed

        return factors
