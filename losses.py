import torch
from torch.nn import MSELoss

_EPSILON = 1e-6

def CIF_K_tau(first_hitting_time, event_k, tte, data_length, MAX_LENGTH):
    #last measurement is on index "data_length - 1"
    #event_time is on index "data_length"
    amount_of_events = first_hitting_time.size(0)//MAX_LENGTH
    numerator = torch.sum(first_hitting_time.view(amount_of_events, MAX_LENGTH)[event_k][data_length - 1:tte])
    denomenator = 1 - torch.sum(first_hitting_time.view(amount_of_events, MAX_LENGTH)[:,:data_length - 1])
    #print("numerator=", numerator)
    #print("denomenator=", denomenator)
    return numerator/(denomenator + _EPSILON)

def CIF_K(first_hitting_time, event_k, data_length, MAX_LENGTH):
    #This would all be more optimal to do in batches, but we're debugging, so we keep it simple, optimise later
    cif_k = torch.zeros(MAX_LENGTH)
    
    for i in range(MAX_LENGTH):
        cif_k[i] = CIF_K_tau(first_hitting_time, event_k, i+1, data_length, MAX_LENGTH)

    return cif_k

def CIF(first_hitting_time, data_length, MAX_LENGTH):
    amount_of_events = first_hitting_time.size(0)//MAX_LENGTH
    cif = torch.zeros(amount_of_events, MAX_LENGTH)
    
    for event in range(amount_of_events):
        cif[event] = CIF_K(first_hitting_time, event, data_length, MAX_LENGTH)
        #print("CIF_K=", CIF_K(first_hitting_time, event, data_length, MAX_LENGTH))

    return cif

def eta(a,b, sigma):
    return torch.exp((-1)*(a-b)/sigma)

def loss_1_batch(first_hitting_time_batch, event_batch, batch_tte, batch_data_length, MAX_LENGTH, device='cpu'):
    sum = torch.zeros(1).to(device)
    amount_of_events = first_hitting_time_batch.size(1)//MAX_LENGTH
    
    for idx, first_hitting_time in enumerate(first_hitting_time_batch):
        event = int(event_batch[idx].item())
        tte = int(batch_tte[idx].item())
        data_length = int(batch_data_length[idx].item())

        #For all samples that experience an event
        if event == 0 or event == 1 or event == 2:
            numerator = first_hitting_time.view(amount_of_events, MAX_LENGTH)[event, tte-1]
            denomenator = 1 - torch.sum(first_hitting_time.view(amount_of_events, MAX_LENGTH)[:,:data_length - 1])
            test = torch.log((numerator/(denomenator + _EPSILON)) + _EPSILON)
            if torch.isnan(test).any():
                print("first NAN is in sample that sees event")
            sum -= test
            
        #For all samples that experience no event (censoring)
        else:
            #we don't know which event the subject will experience, but we know the 
            # subject didn't experience any event before the hitting time
            cif = torch.sum(CIF(first_hitting_time, data_length, MAX_LENGTH)[:,data_length-2]) #Waarom is dit -2?
            #print("data_length=", data_length)
            #print("CIF=", CIF(first_hitting_time, data_length, MAX_LENGTH))
            test = torch.log(1 - cif + _EPSILON)
            if torch.isnan(test).any():
                print("first NAN is in sample that is censored")
            sum -= test

    return sum

def loss_2_batch(first_hitting_time_batch, event_batch, batch_tte, batch_data_length, num_events, max_length, sigma, device='cpu'):
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
            cif_batch[i] = CIF_K(first_hitting_time, event, batch_data_length[i], max_length)

        # get the CIF where our event equals the one we're considering
        all_cif_with_correct_event = cif_batch[event_batch.view(batch_size) == event]
        all_tte_with_correct_event = batch_tte[event_batch.view(batch_size) == event]
        
        # compare the selected CIF's with every CIF that has a hitting time after our selected one
        for cif_with_correct_event, tte_with_correct_event in zip(all_cif_with_correct_event, all_tte_with_correct_event):
            
            for cif, tte in zip(cif_batch, batch_tte):
                if tte_with_correct_event < tte:
                    total_ranking_loss += eta(cif_with_correct_event[tte_with_correct_event.long() - 1] , cif[tte_with_correct_event.long() - 1], sigma)
            

    return total_ranking_loss/batch_size

def loss_3_batch(encoder_output_batch, input_batch):
    mse_loss = MSELoss(reduction="mean")
    return mse_loss(encoder_output_batch, input_batch.detach()[:,1:])