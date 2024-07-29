
import torch.nn as nn


class GRUCDEFunc(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_hidden_size, n_layers):
        super(GRUCDEFunc, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_hidden_size = hidden_hidden_size
        self.n_layers = n_layers

        # alpha
        self.gru_cell_alpha = nn.GRUCell(hidden_size, hidden_hidden_size)
        self.gru_cells_alpha = nn.ModuleList(nn.GRUCell(hidden_hidden_size, hidden_hidden_size)
                                             for _ in range(n_layers - 1))
        self.fc_alpha = nn.Linear(hidden_hidden_size, input_size * hidden_size)

        # beta
        self.gru_cell_beta = nn.GRUCell(hidden_size, hidden_hidden_size)
        self.gru_cells_beta = nn.ModuleList(nn.GRUCell(hidden_hidden_size, hidden_hidden_size)
                                            for _ in range(n_layers - 1))
        self.fc_beta = nn.Linear(hidden_hidden_size, input_size * hidden_size)

    def forward(self, t, gamma):
        alpha, beta = gamma

        # alpha
        alpha = self.gru_cell_alpha(alpha)
        alpha = alpha.relu()
        for gru_cell in self.gru_cells_alpha:
            alpha = gru_cell(alpha)
            alpha = alpha.relu()
        alpha = self.fc_alpha(alpha)

        # beta
        beta = self.gru_cell_beta(beta)
        beta = beta.relu()
        for gru_cell in self.gru_cells_beta:
            beta = gru_cell(beta)
            beta = beta.relu()
        beta = self.fc_beta(beta)

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
