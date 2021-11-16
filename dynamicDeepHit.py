import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu, softmax

_EPSILON = 1e-08
length = 30
num_covariates = 3  # This is not customisable
batch_size = 16

MAX_LENGTH = 30

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
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

    def forward(self, last_measurement, encoder_hidden_vector):
        #TODO Here in the decoder, we abuse the batch system
        # Ik heb het verkeerd gebruikt, nu mapt het de sequence naar 30 waarden per hidden state, wat niet juist is...
        input_for_attention = torch.cat((last_measurement.repeat(self.max_length,1), encoder_hidden_vector), 1)
        importance = self.attn(input_for_attention)
        attn_weights = softmax(importance, dim=1)

        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_hidden_vector.unsqueeze(0))

        return context_vector, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)