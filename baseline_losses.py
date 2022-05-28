import torch

_EPSILON = 1e-6

def CIF_K(first_hitting_time, event_k, MAX_LENGTH):
    #This would all be more optimal to do in batches, but we're debugging, so we keep it simple
    return torch.cumsum(first_hitting_time[event_k*MAX_LENGTH:(event_k+1)*MAX_LENGTH], dim=0)

def CIF(first_hitting_time, MAX_LENGTH):
    amount_of_events = first_hitting_time.size(0)//MAX_LENGTH
    return torch.cumsum(first_hitting_time.view(amount_of_events, MAX_LENGTH), dim=1)

def eta(a,b, sigma):
    return torch.exp((-1)*(a-b)/sigma)

def loss_1_batch(first_hitting_time_batch, event_batch, batch_tte, MAX_LENGTH, device="cpu"):
    sum = torch.zeros(1).to(device)
    amount_of_events = first_hitting_time.size(0)//MAX_LENGTH

    for idx, first_hitting_time in enumerate(first_hitting_time_batch):
        event = int(event_batch[idx].item())
        tte = int(batch_tte[idx].item())

        #For all samples that experience an event
        if event == 0 or event == 1 or event == 2:
            numerator = first_hitting_time.view(amount_of_events, MAX_LENGTH)[event, tte-1]
            sum -= torch.log(numerator + _EPSILON)

            
            
        #For all samples that experience no event (censoring)
        else:
            #we don't know which event the subject will experience, but we know the 
            # subject didn't experience any event before the hitting time
            numerator = torch.sum(CIF(first_hitting_time, MAX_LENGTH)[:,tte-2])
            sum -= torch.log(1 - numerator + _EPSILON)

    return sum

def loss_2_batch(first_hitting_time_batch, event_batch, batch_tte, num_events, max_length, sigma, device="cpu"):
    """
    concordance loss

    define pair (i,j) an acceptable pair for event k, if subject i experiences event k at time s^i, 
    while subject j did not experience any event at time s^i

    TODO: test grondig
    """
    batch_size = event_batch.size(0)
    total_ranking_loss = torch.zeros(1).to(device)

    if batch_size <= 1:
        return total_ranking_loss
    
    # iterate over every possible event
    for event in range(num_events):

        # get the cummulative sum of the batch for the considered event
        cif_batch = torch.zeros(batch_size, max_length).to(device)
        for i, first_hitting_time in enumerate(first_hitting_time_batch):
            cif_batch[i] = CIF_K(first_hitting_time, event, max_length)


        # get the CIF where our event equals the one we're considering
        all_cif_with_correct_event = cif_batch[event_batch.view(batch_size) == event]
        all_tte_with_correct_event = batch_tte[event_batch.view(batch_size) == event]
        
        # compare the selected CIF's with every CIF that has a hitting time after our selected one
        for cif_with_correct_event, tte_with_correct_event in zip(all_cif_with_correct_event, all_tte_with_correct_event):
            
            for cif, tte in zip(cif_batch, batch_tte):
                if tte_with_correct_event < tte:
                    total_ranking_loss += eta(cif_with_correct_event[tte_with_correct_event.long() - 1] , cif[tte_with_correct_event.long() - 1], sigma)
            
    return total_ranking_loss/batch_size