import matplotlib.pyplot as plt
import torch
from losses import CIF_K

#CIF_K(first_hitting_time, event_k, data_length, MAX_LENGTH):

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
    