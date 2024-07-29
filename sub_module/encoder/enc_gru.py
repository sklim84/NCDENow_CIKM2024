import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device):
        super(GRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device

        self.fc = nn.Linear(self.input_size, self.hidden_size)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=n_layers,
                          batch_first=True)

        self.fc_z = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, factors):

        h_proj = F.leaky_relu(self.fc(x))
        h_init = self.init_hidden(batch_size=x.size(0))
        output, h = self.gru(h_proj, h_init)

        h = h[-1]

        z = self.fc_z(h)

        return z

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h_init = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device)
        return h_init
