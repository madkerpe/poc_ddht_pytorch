import torch

_EPSILON = 1e-6


def loss_1_batch(first_hitting_time_batch, event_batch, batch_tte, MAX_LENGTH):
    amount_of_events = first_hitting_time_batch.size(1)//MAX_LENGTH
    sum = 0
    
    
    for idx, first_hitting_time in enumerate(first_hitting_time_batch):
        event = int(event_batch[idx].item())
        tte = int(batch_tte[idx].item())

        #For all samples that experience an event
        if event == 0 or event == 1:
            numerator = first_hitting_time.view(amount_of_events, MAX_LENGTH)[event, tte-1]
            sum -= torch.log(numerator + _EPSILON)

        elif event == 2:
            numerator = first_hitting_time.view(amount_of_events, MAX_LENGTH)[event, tte-1]
            sum -= torch.log(numerator + _EPSILON)

            
        #For all samples that experience no event (censoring)
        else:
            #we don't know which event the subject will experience, but we know the 
            # subject didn't experience any event before the hitting time
            numerator = torch.sum(first_hitting_time.view(amount_of_events, MAX_LENGTH)[:,tte-2]) #Ik heb het vermoeden dat dit eigenlijk weg mag, omdat dit nu gewoon een restant is van continu naar discreet
            sum -= torch.log(1 - numerator + _EPSILON)


    return sum

def eta(a,b, sigma):
    return torch.exp((-1)*(a-b)/sigma)

#LOSS_2_AMPLIFIER*loss_2_batch(first_hitting_time_batch, batch_event, batch_data_length, NUM_CAUSES, MAX_LENGTH, SIGMA, DEVICE)
def loss_2_batch(first_hitting_time_batch, event_batch, batch_tte, num_events, max_length, sigma, device):
    """
    concordance loss

    define pair (i,j) an acceptable pair for event k, if subject i experiences event k at time s^i, 
    while subject j did not experience any event at time s^i

    TODO: test grondig
    """
    batch_size = event_batch.size(0)

    if batch_size <= 1:
        return torch.zeros(1).to(device)

    total_ranking_loss = torch.zeros(1).to(device)
    # iterate over every possible event
    for event in range(num_events):

        # get the cummulative sum of the batch for the considered event
        cif_batch = torch.cumsum(first_hitting_time_batch[:,event*max_length:(event+1)*max_length], dim=1)

        # get the CIF where our event equals the one we're considering
        all_cif_with_correct_event = cif_batch[event_batch.view(batch_size) == event]
        all_tte_with_correct_event = batch_tte[event_batch.view(batch_size) == event]
        
        # compare the selected CIF's with every CIF that has a hitting time after our selected one
        for cif_with_correct_event, tte_with_correct_event in zip(all_cif_with_correct_event, all_tte_with_correct_event):
            
            for cif, tte in zip(cif_batch, batch_tte):
                if tte_with_correct_event < tte:
                    total_ranking_loss += eta(cif_with_correct_event[tte_with_correct_event.long() - 1] , cif[tte_with_correct_event.long() - 1], sigma)
            

    return total_ranking_loss/batch_size