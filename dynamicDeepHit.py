import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu, softmax, tanh

_EPSILON = 1e-08
length = 30
num_covariates = 3  # This is not customisable
batch_size = 16

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = GRU(input_size, hidden_size, batch_first=True)
        self.fc_layer_1 = Linear(hidden_size, hidden_size//2)
        self.fc_layer_2 = Linear(hidden_size//2, hidden_size//4)
        self.fc_layer_3 = Linear(hidden_size//4, input_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = relu(self.fc_layer_1(output))
        output = relu(self.fc_layer_2(output))
        output = self.fc_layer_3(output)

        return output, hidden

    def initHidden(self, device='cpu'):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.attn = Linear(self.hidden_size + output_size, 1)

    def forward(self, last_measurement, encoder_hidden_vector):
        #TODO Here in the decoder, we abuse the batch system, this should be reworked
        input_for_attention = torch.cat((last_measurement.repeat(self.max_length,1), encoder_hidden_vector), 1)
        importance = self.attn(input_for_attention)
        attn_weights = softmax(importance, dim=0)

        context_vector = torch.mm(torch.transpose(attn_weights, 0, 1), encoder_hidden_vector)

        return context_vector

class SharedSubnetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_length, device):
        super(SharedSubnetwork, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, max_length)

        self.encoder_hidden_vector = torch.zeros(max_length, self.encoder.hidden_size, device=device)
        self.encoder_output_vector = torch.zeros(max_length, output_size, device=device)
        self.encoder_hidden = self.encoder.initHidden(device)

    def forward(self, batch, time_to_event):
        input_length = batch.size(1)

        for ei in range(input_length):
            encoder_input = batch[0][ei].view(1,1,-1)
            encoder_output, self.encoder_hidden = self.encoder(encoder_input, self.encoder_hidden)
            self.encoder_hidden_vector[ei] = self.encoder_hidden[0,0].clone()
            self.encoder_output_vector[ei] = encoder_output.clone()


        #last_measurement_index = int(time_to_event.item())
        last_measurement = batch[0][ei]

        context_vector = self.decoder(last_measurement, self.encoder_hidden_vector)

        return context_vector, self.encoder_output_vector
