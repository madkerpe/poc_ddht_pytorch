import matplotlib.pyplot as plt
import torch
from losses import CIF_K
from baseline_losses import CIF_K as CIF_K_baseline

def plot_fht(first_hitting_time, MAX_LENGTH):
    fig, (ax0, ax1, ax2) = plt.subplots(3)
    ax0.bar([i for i in range(MAX_LENGTH)], first_hitting_time[:MAX_LENGTH].cpu().detach().numpy())
    ax1.bar([i for i in range(MAX_LENGTH)], first_hitting_time[MAX_LENGTH:2*MAX_LENGTH].cpu().detach().numpy())
    ax2.bar([i for i in range(MAX_LENGTH)], first_hitting_time[2*MAX_LENGTH:].cpu().detach().numpy())
    ax0.set_title("prepay")
    ax1.set_title("default")
    ax2.set_title("full repay")
    ax0.set_ylim([0,1]);
    ax1.set_ylim([0,1]);
    ax2.set_ylim([0,1]);

def plot_cif(first_hitting_time, data_length, MAX_LENGTH):
    fig, (ax0, ax1, ax2) = plt.subplots(3)
    ax0.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 0, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax1.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 1, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax2.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 2, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax0.set_title("prepay")
    ax1.set_title("default")
    ax2.set_title("full repay")
    ax0.set_ylim([0,1]);
    ax1.set_ylim([0,1]);
    ax2.set_ylim([0,1]);

    

def plot_fht_and_cif(first_hitting_time, data_length, MAX_LENGTH):
    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2,3,figsize=(10,5))

    ax00.bar([i for i in range(MAX_LENGTH)], first_hitting_time[:MAX_LENGTH].cpu().detach().numpy())
    ax01.bar([i for i in range(MAX_LENGTH)], first_hitting_time[MAX_LENGTH:2*MAX_LENGTH].cpu().detach().numpy())
    ax02.bar([i for i in range(MAX_LENGTH)], first_hitting_time[2*MAX_LENGTH:].cpu().detach().numpy())
    ax00.set_title("prepay")
    ax01.set_title("default")
    ax02.set_title("full repay")
    ax00.set_ylim([0,1]);
    ax01.set_ylim([0,1]);
    ax02.set_ylim([0,1]);

    ax10.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 0, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax11.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 1, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax12.bar([i for i in range(MAX_LENGTH)], CIF_K(first_hitting_time, 2, data_length, MAX_LENGTH).cpu().detach().numpy())
    ax10.set_ylim([0,1]);
    ax11.set_ylim([0,1]);
    ax12.set_ylim([0,1]);

def plot_fht_and_cif_baseline(first_hitting_time, MAX_LENGTH):
    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2,3,figsize=(10,5))

    ax00.bar([i for i in range(MAX_LENGTH)], first_hitting_time[:MAX_LENGTH].cpu().detach().numpy())
    ax01.bar([i for i in range(MAX_LENGTH)], first_hitting_time[MAX_LENGTH:2*MAX_LENGTH].cpu().detach().numpy())
    ax02.bar([i for i in range(MAX_LENGTH)], first_hitting_time[2*MAX_LENGTH:].cpu().detach().numpy())
    ax00.set_title("prepay")
    ax01.set_title("default")
    ax02.set_title("full repay")
    ax00.set_ylim([0,1]);
    ax01.set_ylim([0,1]);
    ax02.set_ylim([0,1]);

    ax10.bar([i for i in range(MAX_LENGTH)], CIF_K_baseline(first_hitting_time, 0, MAX_LENGTH).cpu().detach().numpy())
    ax11.bar([i for i in range(MAX_LENGTH)], CIF_K_baseline(first_hitting_time, 1, MAX_LENGTH).cpu().detach().numpy())
    ax12.bar([i for i in range(MAX_LENGTH)], CIF_K_baseline(first_hitting_time, 2, MAX_LENGTH).cpu().detach().numpy())
    ax10.set_ylim([0,1]);
    ax11.set_ylim([0,1]);
    ax12.set_ylim([0,1]);
    
def plot_gamma(gamma, y_lim=None):
    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(10,5))

    input_size = gamma.shape[1]

    ax0.bar([i for i in range(input_size)], gamma[0].cpu().detach().numpy())
    ax1.bar([i for i in range(input_size)], gamma[1].cpu().detach().numpy())
    ax2.bar([i for i in range(input_size)], gamma[2].cpu().detach().numpy())
    
    ax0.set_title("indicators prepay event")
    ax1.set_title("indicators default event")
    ax2.set_title("indicators full repay event")
    
    if y_lim is not None:
        ax0.set_ylim(y_lim);
        ax1.set_ylim(y_lim);
        ax2.set_ylim(y_lim);

def plot_attention_weights_batch(attention_weights_batch, data_length_batch, normalised=True, y_lim=None):
    fig, ax = plt.subplots(1,figsize=(10,5))
    
    attention_weights_batch = attention_weights_batch.cpu()
    data_length_batch = data_length_batch.cpu()


    batch_size = attention_weights_batch.shape[0]
    attention_weights_length = attention_weights_batch.shape[1]
    normalization_vector = torch.zeros(attention_weights_length)

    attention_weights = (1.0/batch_size)*torch.sum(attention_weights_batch, dim=0)

    if normalised:
        for length in data_length_batch:
            normalization_vector[:length] += torch.ones(length)

        normalization_vector = (1/batch_size)*normalization_vector
        attention_weights = torch.div(attention_weights,normalization_vector)

    else: 
        print("WARNING: This function isn't normalised in length of samples")

    ax.bar([i for i in range(attention_weights_length)], attention_weights.cpu().detach().numpy())

    if y_lim is not None:
        ax.set_ylim([0,y_lim]);

def plot_attention_weights(attention_weights):
    fig, ax = plt.subplots(1,figsize=(10,5))

    attention_weights_length = attention_weights.shape[0]

    ax.bar([i for i in range(attention_weights_length)], attention_weights.cpu().detach().numpy())
    ax.set_ylim([0,1]);