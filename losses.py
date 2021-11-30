import torch
from torch.nn import MSELoss, CrossEntropyLoss

_EPSILON = 1e-10

def loss_1(first_hitting_time, event, time_of_event, max_length):
    """
    negative log likelihoo loss, assuming the batch dimension on the first dimension
    """
    event_observation_index = (event*max_length + time_of_event).long()
    event_observation = torch.log(first_hitting_time[:,event_observation_index] + _EPSILON)
    #kijk eens of hier niet nog expliciet in moet dat de rest nul moet zijn

    return (-1)*torch.sum(event_observation)

def loss_3(encoder_output_vector, batch):
    mse_loss = MSELoss(reduction="sum")
    return mse_loss(encoder_output_vector[:-1], batch.detach().squeeze(0)[1:])

def loss_1_batch(first_hitting_time_batch, event_batch, time_to_event_batch, MAX_LENGTH):
    """
    TODO: negative log likelihood loss, assuming the batch dimension on the first dimension
    """
    batch_size = first_hitting_time_batch.size(0)
    event_observation_index = (event_batch*MAX_LENGTH + time_to_event_batch).view(batch_size).long()
    #event_observation = torch.log(first_hitting_time[:,event_observation_index] + _EPSILON)
    #kijk eens of hier niet nog expliciet in moet dat de rest nul moet zijn

    #we gebruiken nu tijdelijk gewoon de crossentropyloss, dan krijgen we een categorical distrib.
    crl = CrossEntropyLoss(reduction="sum")
    return crl(first_hitting_time_batch, event_observation_index)

def eta(a,b, sigma):
    sigma = 1
    return torch.exp((-1)*(a-b)/sigma)

def loss_2_batch(first_hitting_time_batch, event_batch, time_to_event_batch, num_events, max_length, sigma):
    """
    concordance loss

    define pair (i,j) an acceptable pair for event k, if subject i experiences event k at time s^i, 
    while subject j did not experience any event at time s^i
    """
    batch_size = first_hitting_time_batch.size(0)
    total_ranking_loss = 0
    # iterate over every possible event
    for event in range(num_events):

        # get the cummulative sum of the batch for the considered event
        cif_batch = torch.cumsum(first_hitting_time_batch[:,event*max_length:(event+1)*max_length], dim=1)

        # get the CIF where our event equals the one we're considering
        all_cif_with_correct_event = cif_batch[event_batch.view(batch_size) == event]
        all_tte_with_correct_event = time_to_event_batch[event_batch.view(batch_size) == event]
        
        # compare the selected CIF's with every CIF that has a hitting time after our selected one
        for cif_with_correct_event, tte_with_correct_event in zip(all_cif_with_correct_event, all_tte_with_correct_event):
            
            for cif, tte in zip(cif_batch, time_to_event_batch):
                if tte_with_correct_event < tte:
                    total_ranking_loss += eta(cif_with_correct_event[tte_with_correct_event.long()] , cif[tte_with_correct_event.long()], sigma)
            

    return total_ranking_loss






# def ranking_loss(cif, t, e, sigma):
#     """
#         Penalize wrong ordering of probability
#         Equivalent to a C Index
#         This function is used to penalize wrong ordering in the survival prediction
#     """
#     loss = 0
#     # Data ordered by time
#     for k, cifk in enumerate(cif):
#         for ci, ti in zip(cifk[e-1 == k], t[e-1 == k]):
#             # For all events: all patients that didn't experience event before
#             # must have a lower risk for that cause
#             if torch.sum(t > ti) > 0:
#                 # TODO: When data are sorted in time -> wan we make it even faster ?
#                 loss += torch.mean(torch.exp((cifk[t > ti][:, ti] - ci[ti])) / sigma)
    

def loss_3_batch(encoder_output_batch, input_batch):
    mse_loss = MSELoss(reduction="sum")
    return mse_loss(encoder_output_batch[:,:-1], input_batch.detach()[:,1:])