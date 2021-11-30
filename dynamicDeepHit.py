from typing import ForwardRef
import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu, softmax, tanh, sigmoid

num_covariates = 3  # This is not customisable

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, fc_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = GRU(input_size, hidden_size, batch_first=True)
        self.fc_layer_1 = Linear(hidden_size, fc_size)
        self.fc_layer_2 = Linear(fc_size, fc_size//2)
        self.fc_layer_3 = Linear(fc_size//2, input_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = relu(self.fc_layer_1(output))
        output = relu(self.fc_layer_2(output))
        output = self.fc_layer_3(output)

        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnNetwork(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(AttnNetwork, self).__init__()
        self.max_length = max_length
        self.layer_1 = Linear(hidden_size + output_size, 1)

    def forward(self, last_measurement, encoder_hidden_vector):
        #TODO Here in the decoder, we abuse the batch system, this should be reworked
        input_for_attention = torch.cat((last_measurement.repeat(self.max_length,1), encoder_hidden_vector), 1)
        importance = self.layer_1(input_for_attention)
        attn_weights = softmax(importance, dim=0)

        return attn_weights

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.attn = AttnNetwork(hidden_size, output_size, max_length)

    def forward(self, last_measurement, encoder_hidden_vector):
        attn_weights = self.attn(last_measurement, encoder_hidden_vector)
        context_vector = torch.mm(torch.transpose(attn_weights, 0, 1), encoder_hidden_vector)

        return context_vector

class RegressionNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionNetwork, self).__init__()
        self.layer1 = Linear(input_size, hidden_size//2)
        self.layer2 = Linear(hidden_size//2, 1)
    
    def forward(self, input):
        output = relu(self.layer1(input))
        output = torch.sigmoid(self.layer2(output))

        return output


class SharedSubnetwork(torch.nn.Module):
    def __init__(self):
        super(SharedSubnetwork, self).__init__()
        pass

    def forward(self):
        pass

class CauseSpecificSubnetwork(torch.nn.Module):
    def __init__(self, hidden_size, input_size, max_length, num_causes):
        super(CauseSpecificSubnetwork, self).__init__()
        self.layer1 = Linear(hidden_size + input_size, 2*hidden_size)
        self.layer2 = Linear(2*hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, hidden_size//2)
        self.layer4 = Linear(hidden_size//2, max_length*num_causes)

    def forward(self, context_vector, last_measurement):
        output = relu(self.layer1(torch.cat((context_vector, last_measurement.unsqueeze(0)), dim=1)))
        output = relu(self.layer2(output))
        output = relu(self.layer3(output))
        output = softmax(self.layer4(output), dim=1)

        return output