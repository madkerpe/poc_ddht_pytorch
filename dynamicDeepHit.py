import torch
from torch.nn import Linear, GRU, Dropout
from torch.nn.functional import relu, softmax
from torch.nn.init import xavier_uniform_

class EncoderRNN(torch.nn.Module):
    def __init__(self, covariate_input_size, rnn_state_size, encoder_fc_size, covariate_output_size):
        super(EncoderRNN, self).__init__()
        self.rnn_state_size = rnn_state_size
        self.input_size = covariate_input_size
        self.gru = GRU(covariate_input_size, rnn_state_size, batch_first=True)
        self.fc_layer_1 = Linear(rnn_state_size, encoder_fc_size)
        self.fc_dropout_1 = Dropout(0.6)
        self.fc_layer_2 = Linear(encoder_fc_size, covariate_output_size)

        xavier_uniform_(self.fc_layer_1.weight)
        xavier_uniform_(self.fc_layer_2.weight)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = relu(self.fc_layer_1(output))
        output = self.fc_dropout_1(output)
        output = self.fc_layer_2(output)

        return output, hidden

    def initHidden(self, device):
        return torch.rand(1, 1, self.rnn_state_size, device=device)

class AttnNetwork(torch.nn.Module):
    def __init__(self, rnn_state_size, attention_fc_size, covariate_input_size):
        super(AttnNetwork, self).__init__()
        self.input_size_for_attention = rnn_state_size + covariate_input_size

        self.fc_layer_1 = Linear(self.input_size_for_attention, attention_fc_size)
        self.fc_layer_2 = Linear(attention_fc_size, 1)

        xavier_uniform_(self.fc_layer_1.weight)
        xavier_uniform_(self.fc_layer_2.weight)

    def forward(self, last_measurement, rnn_hidden_state_vector):
        #TODO Here in the decoder, we abuse the batch system, this should be reworked
        input_for_attention = torch.cat((last_measurement.repeat(rnn_hidden_state_vector.size(0),1), rnn_hidden_state_vector), 1)
        
        importance = relu(self.fc_layer_1(input_for_attention))
        importance = self.fc_layer_2(importance)
        attn_weights = softmax(importance, dim=0)

        return attn_weights

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, rnn_state_size, covariate_input_size, attention_fc_size):
        super(AttnDecoderRNN, self).__init__()
        self.rnn_state_size = rnn_state_size
        self.covariate_input_size = covariate_input_size
        self.attn = AttnNetwork(rnn_state_size, attention_fc_size, covariate_input_size)

    def forward(self, last_measurement, rnn_state_vector):
        attn_weights = self.attn(last_measurement, rnn_state_vector)
        context_vector = torch.mm(torch.transpose(attn_weights, 0, 1), rnn_state_vector)

        return context_vector, attn_weights

class CauseSpecificSubnetwork(torch.nn.Module):
    def __init__(self, context_size, covariate_input_size, cause_fc_size, max_length, num_causes):
        super(CauseSpecificSubnetwork, self).__init__()
        self.cause_input_size = context_size + covariate_input_size
        self.cause_output_size = max_length*num_causes

        self.fc_layer_1 = Linear(self.cause_input_size, cause_fc_size)
        self.dropout_layer_1 = Dropout(0.6)
        self.fc_layer_2 = Linear(cause_fc_size, cause_fc_size)
        self.dropout_layer_2 = Dropout(0.6)
        self.fc_layer_3 = Linear(cause_fc_size, cause_fc_size)
        self.dropout_layer_3 = Dropout(0.6)
        self.fc_layer_4 = Linear(cause_fc_size, self.cause_output_size)

        xavier_uniform_(self.fc_layer_1.weight)
        xavier_uniform_(self.fc_layer_2.weight)
        xavier_uniform_(self.fc_layer_3.weight)
        xavier_uniform_(self.fc_layer_4.weight)

    def forward(self, context_vector, last_measurement):
        output = relu(self.fc_layer_1(torch.cat((context_vector, last_measurement.unsqueeze(0)), dim=1)))
        output = self.dropout_layer_1(output)
        output = relu(self.fc_layer_2(output))
        output = self.dropout_layer_2(output)
        output = relu(self.fc_layer_3(output))
        output = self.dropout_layer_3(output)
        output = softmax(self.fc_layer_4(output), dim=1)

        return output

class DynamicDeepHit(torch.nn.Module):
    def __init__(self, encoder_network, decoder_network, causes_network, max_length, device):
        super(DynamicDeepHit, self).__init__()
        self.encoder = encoder_network
        self.decoder = decoder_network
        self.causess = causes_network

        self.max_length = max_length
        self.device = device

    def forward(self, input_batch, data_length_batch):
        DEVICE = self.device

        output_batch = torch.zeros((input_batch.size(0), input_batch.size(1) - 1, input_batch.size(2)), device=DEVICE)
        #(b,l-1,d)
        first_hitting_time_batch = torch.zeros((input_batch.size(0), self.causess.cause_output_size), device=DEVICE)
        #(b,l*k)
        attn_weights_batch = torch.zeros((input_batch.size(0), self.max_length - 1), device=DEVICE)

        for idx, data in enumerate(zip(input_batch, data_length_batch)):

            sample, data_length = data
            #(l,d)

            data_length = int(data_length.item())
            observed_length = sample.size(0)
            #=l-1

            encoder_hidden_vector = torch.zeros(observed_length - 1, self.encoder.rnn_state_size, device=DEVICE)
            #(l-1,h)
            encoder_output_vector = torch.zeros(observed_length - 1, self.encoder.input_size, device=DEVICE)
            #(l-1,d)
            encoder_hidden = self.encoder.initHidden(device=DEVICE)
            #(1,1,h)

            last_measurement = sample[data_length - 1]
            #(d)

            #TODO Batch optimalisation should be made here, unroll time in parallel
            #Push every timestep except the last measurement through the encoder
            for ei in range(data_length - 1):
                encoder_input = sample[ei].view(1,1,-1)
                #(1,1,d)
                encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
                #(1,1,d), (1,1,h)
                encoder_hidden_vector[ei] = encoder_hidden[0,0]
                encoder_output_vector[ei] = encoder_output

            output_batch[idx] = encoder_output_vector
        
            #TODO Batch optimalisation can be made here
            context_vector, attn_weights = self.decoder(last_measurement, encoder_hidden_vector)
            attn_weights_batch[idx] = attn_weights.view(self.max_length - 1)
            first_hitting_time_batch[idx] = self.causess(context_vector, last_measurement)

        return output_batch, first_hitting_time_batch, attn_weights_batch