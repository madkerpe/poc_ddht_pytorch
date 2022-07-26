import torch
from torch.nn import Linear
from torch.nn.functional import relu, softmax

num_covariates = 3  # This is not customisable

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size_encoder, context_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.fc_layer_1 = Linear(input_size, hidden_size_encoder//2)
        self.fc_layer_2 = Linear(hidden_size_encoder//2, hidden_size_encoder)
        self.fc_layer_3 = Linear(hidden_size_encoder, context_size)

    def forward(self, input):
        output = relu(self.fc_layer_1(input))
        output = relu(self.fc_layer_2(output))
        output = self.fc_layer_3(output)

        return output

class CauseSpecificSubnetwork(torch.nn.Module):
    def __init__(self, context_size, hidden_cause_size, input_size, max_length, num_causes):
        super(CauseSpecificSubnetwork, self).__init__()
        self.input_size = context_size + input_size
        self.output_size = max_length*num_causes
        self.layer1 = Linear(self.input_size, 2*hidden_cause_size)
        self.layer2 = Linear(2*hidden_cause_size, hidden_cause_size)
        self.layer3 = Linear(hidden_cause_size, hidden_cause_size//2)
        self.layer4 = Linear(hidden_cause_size//2, self.output_size)

    def forward(self, context_vector, last_measurement):

        output = relu(self.layer1(torch.cat((context_vector, last_measurement), dim=1)))
        output = relu(self.layer2(output))
        output = relu(self.layer3(output))
        output = softmax(self.layer4(output), dim=1)

        return output

class DeepHit(torch.nn.Module):
    def __init__(self, encoder_network, causes_network, device):
        super(DeepHit, self).__init__()
        self.encoder = encoder_network
        self.causess = causes_network

        self.device = device

    def forward(self, input_batch, time_to_event_batch):
        DEVICE = self.device

        first_hitting_time_batch = torch.zeros((input_batch.size(0), self.causess.output_size), device=DEVICE)
        last_measurement_batch = torch.zeros((input_batch.size(0), self.encoder.input_size), device=DEVICE)

        for idx, data in enumerate(zip(input_batch, time_to_event_batch)):
            sample, tte = data
            tte = int(tte.item())
            last_measurement_batch[idx] = sample[0]
            # if we're working with the POC data, we need to add "= sample[tte - 1]" to the end of the line above
            # the reason is that when POC'in deephit, i was too lazy to write another dataset, so I just took the 
            # dynamic POC dataset and took the last measurement here

        context_vector_batch = self.encoder(last_measurement_batch)
        first_hitting_time_batch = self.causess(context_vector_batch, last_measurement_batch)

        return first_hitting_time_batch