import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu, softmax

_EPSILON = 1e-08
length = 30
num_covariates = 3  # This is not customisable
batch_size = 16

MAX_LENGTH = 30

class SharedSubnetwork(torch.nn.Module):
    def __init__(self, num_covariates=3, hidden_states=32):
        torch.nn.Module.__init__(self)
        self.gru = GRU(input_size=num_covariates, hidden_size=hidden_states, batch_first=True, num_layers=1)
        self.linear = Linear(hidden_states, num_covariates)

    def forward(self, x):
        gru_output, gru_hidden = self.gru(x)
        h = self.linear(gru_output)
        y = relu(h)
        
        return y, gru_hidden

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.fc_layer = Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fc_layer(output)
        output = relu(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.attn = Linear(self.hidden_size + output_size, self.max_length)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = self.attn(torch.cat((input[0], hidden[0]), 1))
        attn_weights = softmax(attn_weights, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        return context_vector

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)