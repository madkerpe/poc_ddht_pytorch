import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu

_EPSILON = 1e-08
length = 30
num_covariates = 3  # This is not customisable
batch_size = 16

class sharedSubnetwork(torch.nn.Module):
    def __init__(self, num_covariates=3, hidden_states=32):
        torch.nn.Module.__init__(self)
        self.gru = GRU(input_size=num_covariates, hidden_size=hidden_states, batch_first=True, num_layers=1)
        self.linear = Linear(hidden_states, num_covariates)

    def forward(self, x):
        gru_output, gru_hidden = self.gru(x)
        h = self.linear(gru_output)
        y = relu(h)
        
        return y, gru_hidden
