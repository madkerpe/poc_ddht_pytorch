import torch

_EPSILON = 1e-10

def loss_1(first_hitting_time, event, time_of_event, max_length):
    """
    negative log likelihoof loss, assuming the batch dimension on the first dimension
    """
    event_observation_index = (event*max_length + time_of_event).long()
    event_observation = torch.log(first_hitting_time[:,event_observation_index] + _EPSILON)
    #kijk eens of hier niet nog expliciet in moet dat de rest nul moet zijn

    return (-1)*torch.sum(event_observation)