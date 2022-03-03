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
        return torch.rand(1, 1, self.hidden_size, device=device)

class AttnNetwork(torch.nn.Module):
    def __init__(self, encoder_hidden_size, attention_hidden_size, output_size, max_length):
        super(AttnNetwork, self).__init__()
        self.max_length = max_length
        self.layer_1 = Linear(encoder_hidden_size + output_size, attention_hidden_size)
        self.layer_2 = Linear(attention_hidden_size, 1)

    def forward(self, last_measurement, encoder_hidden_vector):
        #TODO Here in the decoder, we abuse the batch system, this should be reworked
        input_for_attention = torch.cat((last_measurement.repeat(self.max_length,1), encoder_hidden_vector), 1)
        importance = relu(self.layer_1(input_for_attention))
        importance = self.layer_2(importance)
        attn_weights = softmax(importance, dim=0)

        return attn_weights

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, encoder_hidden_size, attention_hidden_size, output_size, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = encoder_hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.attn = AttnNetwork(encoder_hidden_size, attention_hidden_size, output_size, max_length)

    def forward(self, last_measurement, encoder_hidden_vector):
        attn_weights = self.attn(last_measurement, encoder_hidden_vector)
        context_vector = torch.mm(torch.transpose(attn_weights, 0, 1), encoder_hidden_vector)

        return context_vector

class CauseSpecificSubnetwork(torch.nn.Module):
    def __init__(self, hidden_size, input_size, max_length, num_causes):
        super(CauseSpecificSubnetwork, self).__init__()
        self.input_size = hidden_size + input_size
        self.output_size = max_length*num_causes
        self.layer1 = Linear(self.input_size, 2*hidden_size)
        self.layer2 = Linear(2*hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, hidden_size//2)
        self.layer4 = Linear(hidden_size//2, self.output_size)

    def forward(self, context_vector, last_measurement):
        output = relu(self.layer1(torch.cat((context_vector, last_measurement.unsqueeze(0)), dim=1)))
        output = relu(self.layer2(output))
        output = relu(self.layer3(output))
        output = softmax(self.layer4(output), dim=1)

        return output

class DynamicDeepHit(torch.nn.Module):
    def __init__(self, encoder_network, decoder_network, causes_network, max_length, device):
        super(DynamicDeepHit, self).__init__()
        self.encoder = encoder_network
        self.decoder = decoder_network
        self.causess = causes_network

        self.max_length = max_length
        self.device = device

    def forward(self, input_batch, time_to_event_batch):
        MAX_LENGTH = self.max_length
        DEVICE = self.device

        output_batch = torch.zeros_like(input_batch, device=DEVICE)
        #(b,l,d)
        first_hitting_time_batch = torch.zeros((input_batch.size(0), self.causess.output_size), device=DEVICE)
        #(b,l*k)

        for idx, data in enumerate(zip(input_batch, time_to_event_batch)):

            batch, tte = data
            #(l,d)

            encoder_hidden_vector = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=DEVICE)
            #(l,h)
            encoder_output_vector = torch.zeros(MAX_LENGTH, self.encoder.input_size, device=DEVICE)
            #(l,d)
            encoder_hidden = self.encoder.initHidden(device=DEVICE)
            #(1,1,h)
            
            last_measurement_index = int(tte.item())
            #(1)
            last_measurement = batch[last_measurement_index]
            #(d)

            #TODO Batch optimalisation should be made here, unroll time in parallel
            for ei in range(last_measurement_index):
                encoder_input = batch[ei].view(1,1,-1)
                #(1,1,d)
                encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
                #(1,1,d), (1,1,h)
                encoder_hidden_vector[ei] = encoder_hidden[0,0]
                encoder_output_vector[ei] = encoder_output

            output_batch[idx] = encoder_output_vector
        
            #TODO Batch optimalisation can be made here
            context_vector = self.decoder(last_measurement, encoder_hidden_vector)
            first_hitting_time_batch[idx] = self.causess(context_vector, last_measurement)

        return output_batch, first_hitting_time_batch