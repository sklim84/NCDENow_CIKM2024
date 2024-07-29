import torch
import torch.nn as nn


class MLPCDEFunc(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_hidden_size, n_layers):
        super(MLPCDEFunc, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.hidden_hidden_channels = hidden_hidden_size

        # alpha
        self.linear_in_alpha = nn.Linear(hidden_size, hidden_hidden_size)
        self.linears_alpha = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_size, hidden_hidden_size)
                                                 for _ in range(n_layers - 1))
        self.linear_out_alpha = nn.Linear(hidden_hidden_size, input_size * hidden_size)

        # beta
        self.linear_in_beta = nn.Linear(hidden_size, hidden_hidden_size)
        self.linears_beta = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_size, hidden_hidden_size)
                                                for _ in range(n_layers - 1))
        self.linear_out_beta = nn.Linear(hidden_hidden_size, input_size * hidden_size)

        # factors
        self.linear_in_factors = nn.Linear(hidden_size, hidden_hidden_size)
        self.linears_factors = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_size, hidden_hidden_size)
                                                   for _ in range(n_layers - 1))
        self.linear_out_factors = nn.Linear(hidden_hidden_size, input_size * hidden_size)

    def forward(self, t, gamma):
        alpha, beta = gamma

        # alpha
        alpha = self.linear_in_alpha(alpha)
        alpha = alpha.relu()
        for linear in self.linears_alpha:
            alpha = linear(alpha)
            alpha = alpha.relu()
        alpha = self.linear_out_alpha(alpha)

        # beta
        beta = self.linear_in_beta(beta)
        beta = beta.relu()
        for linear in self.linears_beta:
            beta = linear(beta)
            beta = beta.relu()
        beta = self.linear_out_beta(beta)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        alpha = alpha.tanh()
        beta = beta.tanh()
        # factors = factors.tanh()

        ######################
        # Ignoring the batch dimensions, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        alpha = alpha.view(*alpha.shape[:-1], self.hidden_size, self.input_size)
        beta = beta.view(*beta.shape[:-1], self.hidden_size, self.input_size)

        return alpha, beta
