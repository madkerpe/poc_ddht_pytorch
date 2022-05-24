import matplotlib.pyplot as plt
import torch

def plot_fht(first_hitting_time, MAX_LENGTH):
    fig, (ax0, ax1, ax2) = plt.subplots(3)
    ax0.bar([i for i in range(MAX_LENGTH)], first_hitting_time[:MAX_LENGTH].cpu().detach().numpy())
    ax1.bar([i for i in range(MAX_LENGTH)], first_hitting_time[MAX_LENGTH:2*MAX_LENGTH].cpu().detach().numpy())
    ax2.bar([i for i in range(MAX_LENGTH)], first_hitting_time[2*MAX_LENGTH:].cpu().detach().numpy())
    ax0.set_title("event 0")
    ax1.set_title("event 1")
    ax2.set_title("event 2")
    ax0.set_ylim([0,1]);
    ax1.set_ylim([0,1]);
    ax2.set_ylim([0,1]);

def plot_cif(first_hitting_time, MAX_LENGTH):
    fig, (ax0, ax1, ax2) = plt.subplots(3)
    ax0.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[:MAX_LENGTH], dim=0).cpu().detach().numpy())
    ax1.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[MAX_LENGTH:2*MAX_LENGTH], dim=0).cpu().detach().numpy())
    ax2.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[2*MAX_LENGTH:], dim=0).cpu().detach().numpy())
    ax0.set_title("event 0")
    ax1.set_title("event 1")
    ax2.set_title("event 2")
    ax0.set_ylim([0,1]);
    ax1.set_ylim([0,1]);
    ax2.set_ylim([0,1]);

def plot_fht_and_cif(first_hitting_time, MAX_LENGTH):
    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2,3,figsize=(10,5))

    ax00.bar([i for i in range(MAX_LENGTH)], first_hitting_time[:MAX_LENGTH].cpu().detach().numpy())
    ax01.bar([i for i in range(MAX_LENGTH)], first_hitting_time[MAX_LENGTH:2*MAX_LENGTH].cpu().detach().numpy())
    ax02.bar([i for i in range(MAX_LENGTH)], first_hitting_time[2*MAX_LENGTH:].cpu().detach().numpy())
    ax00.set_title("event 0")
    ax01.set_title("event 1")
    ax02.set_title("event 2")
    ax00.set_ylim([0,1]);
    ax01.set_ylim([0,1]);
    ax02.set_ylim([0,1]);

    ax10.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[:MAX_LENGTH], dim=0).cpu().detach().numpy())
    ax11.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[MAX_LENGTH:2*MAX_LENGTH], dim=0).cpu().detach().numpy())
    ax12.bar([i for i in range(MAX_LENGTH)], torch.cumsum(first_hitting_time[2*MAX_LENGTH:], dim=0).cpu().detach().numpy())
    ax10.set_title("event 0")
    ax11.set_title("event 1")
    ax12.set_title("event 2")
    ax10.set_ylim([0,1]);
    ax11.set_ylim([0,1]);
    ax12.set_ylim([0,1]);
    