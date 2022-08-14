import torch
import numpy as np

_EPSILON = 1e-6

def concordance_index_time_dependent(cif_batch, event_batch, tte_batch, k, t, device='cpu'):
    """
    an instance i experiencing event k_i at time t_i should be assigned
    a higher cummulative probability for event k_i at that time point than an
    instance j that has not experienced any event untill t_i

    cif_batch = CIFs for all the events for every sample in the batch (batch_size x 1 (=num events) x max_length)
    event_batch = the event that each sample in the batch experienced (batch_size x 1)
    tte_batch = the time to event for each sample in the batch (batch_size x 1)
    k = event that we're considering (1)
    t = months after loan was issued (1)

    note that A_k_i_j is defined as Indicator(k_i = k AND t_i <= t_j) 

    TODO: test thoroughly
    """

    batch_size = event_batch.shape[0]
    amount_of_correct_ordered_pairs = torch.zeros(1).to(device)
    amount_of_valid_pairs = torch.zeros(1).to(device)

    if batch_size <= 1:
        return torch.ones(1).to(device)

    # select all samples that experience the considered event

    cif_with_correct_event_batch = cif_batch[event_batch.view(batch_size) == k, k, t]
    tte_with_correct_event_batch = tte_batch[event_batch.view(batch_size) == k]
    all_cif = cif_batch[:, k, t]
    all_tte = tte_batch

    # compare the selected CIF's with every CIF that has a hitting time after our selected one
    for cif_with_correct_event, tte_with_correct_event in zip(cif_with_correct_event_batch, tte_with_correct_event_batch):
        
        for cif, tte in zip(all_cif, all_tte):

            # TODO: this is just blindly following the A_k_i_j definition of blumenstock et al., this could be a mistake,
            #   it could for example be more usefull to compare with "t" from the arguments or something? 
            #     for example: doesn't this introduce a problem if the event always happens at tte = max_length?
            if tte_with_correct_event < tte: # A_k_i_j assuming selection of event is correctly made
                amount_of_valid_pairs += 1
                
                if cif_with_correct_event > cif:
                    amount_of_correct_ordered_pairs += 1

    if amount_of_valid_pairs == 0:
        print("No valid pairs were found for parameters k=%d, t=%d --> returning 0" % (k, t))
        return torch.zeros(1).to(device)

    return amount_of_correct_ordered_pairs/(amount_of_valid_pairs + _EPSILON)

def concordance_index_time_dependent_longitudinal(cif_batch, event_batch, tte_batch, k, t, delta_t, device='cpu'):
    """
    an instance i experiencing event k_i at time t_i should be assigned
    a higher cummulative probability for event k_i at that time point than an
    instance j that has not experienced any event untill t_i

    cif_batch = CIFs for all the events for every sample in the batch (batch_size x 1 (=num events) x max_length)
    event_batch = the event that each sample in the batch experienced (batch_size x 1)
    tte_batch = the time to event for each sample in the batch (batch_size x 1)
    k = event that we're considering (1)
    t = months after loan was issued (1)
    delta_t = evaluation time after t months (1)

    unfortunately, since this has longitudinal we can't evaluate this for the regular DeepHit model
    
    note that A_k_i_j is defined as Indicator(k_i = k AND t_i <= t_j) 
    TODO: test thoroughly
    """

    print("WARNING: this longitudinal concordance doesn't correctly implement the longitudinal dynamic behaviour")

    batch_size = event_batch.shape[0]
    amount_of_correct_ordered_pairs = torch.zeros(1).to(device)
    amount_of_valid_pairs = torch.zeros(1).to(device)

    if batch_size <= 1:
        return torch.ones(1).to(device)

    # select all samples that experience the considered event
    # TODO: The dynamic predition part denotes that we predict at time t + delta_t but only with measurements up untill time t
    #   I don't think this is the case right now, the j'th sample longitudinal data should be shortened to the same length as the i'th sample
    cif_with_correct_event_plus_delta_t_batch = cif_batch[event_batch.view(batch_size) == k, k, t + delta_t]
    tte_with_correct_event_batch = tte_batch[event_batch.view(batch_size) == k]
    all_cif_plus_delta_t = cif_batch[:, k, t + delta_t]
    all_tte = tte_batch

    # compare the selected CIF's with every CIF that has a hitting time after our selected one
    for cif_with_correct_event_plus_delta_t, tte_with_correct_event in zip(cif_with_correct_event_plus_delta_t_batch, tte_with_correct_event_batch):
        
        for cif_plus_delta_t, tte in zip(all_cif_plus_delta_t, all_tte):

            if tte_with_correct_event < tte and tte_with_correct_event < t + delta_t: # tau_i < tau_j and t < t + delta_t
                amount_of_valid_pairs += 1
                
                if cif_with_correct_event_plus_delta_t > cif_plus_delta_t:
                    amount_of_correct_ordered_pairs += 1

    if amount_of_valid_pairs == 0:
        print("No valid pairs were found for parameters k=%d, t=%d, delta_t=%d --> returning 0" % (k, t, delta_t))
        return torch.zeros(1).to(device)

    return amount_of_correct_ordered_pairs/(amount_of_valid_pairs + _EPSILON)