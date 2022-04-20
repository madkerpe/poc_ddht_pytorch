import numpy as np
import torch
import math
import matplotlib.pyplot as plt

min_length = 5
max_length = 12*3
num_covariates = 5
epsilon = 1e-6



def generate_sample():
    """
    data:
        data[:, 0] --> age in months
        data[:, 1] --> original duration in months
        data[:, 2] --> remaining months
        data[:, 3] --> X*sin(age + X)
        data[:, 4] --> Y*cos(age + Y)

    event:
        0 --> full repay
        1 --> prepay
        2 --> default

    """

    # define the latent variables X and Y
    X = torch.rand(1)
    Y = torch.rand(1)

    # deduce the event based on X and the length based on Y
    if X > 0.5:
        event = 0
        length = max_length

    elif X <= 0.25:
        event = 1
        length = max(math.floor(Y*max_length - epsilon), min_length)
        # length at least min_length, length max the max_length minus one

    else:
        event = 2
        length = max(math.floor(Y*max_length - epsilon), min_length)

    # reserve space for a data sample
    data = torch.zeros(length, num_covariates)

    # add some random noise on the start of the sample
    data[:, 0] = torch.arange(0, length, dtype=torch.float32)
    data[:, 1] = length*torch.ones(length)
    data[:, 2] = length - torch.arange(0, length, dtype=torch.float32) - 1
    data[:, 3] = X*torch.sin(torch.arange(0, length, dtype=torch.float32) + X)
    data[:, 4] = Y*torch.cos(torch.arange(0, length, dtype=torch.float32) + Y)

    # define the latent variables Z1 and Z2
    Z1 = torch.rand(1)
    Z2 = torch.rand(1)

    # censoring based on Z1 and Z2
    if Z1 < 0.5:
        # we do left censoring
        left_censor = math.floor(Z1*length)
        data = data[left_censor:]

    if Z2 > 0.5:
        # we do right censoring
        right_censor = math.floor(Z2*length)
        data = data[:right_censor]

    meta = {'X': X, 'Y': Y, 'Z1': Z1, 'Z2': Z2}

    return data, event

def display_sample(sample):
    print("Length is %d, Event is %d" % (sample[0].shape[0], sample[1]))
    plt.plot(np.array(sample[0][:,0]), np.array(sample[0][:,3]))
    plt.plot(np.array(sample[0][:,0]), np.array(sample[0][:,4]))

class PocDataset(torch.utils.data.Dataset):
    def __init__(self, num_cases=1024):
        torch.utils.data.Dataset.__init__(self)
        self.num_cases = num_cases

        self.data = []
        self.event = torch.zeros(num_cases, 1)

        for i in range(num_cases):
            sample = generate_sample()
            self.data.append(sample[0])
            self.event[i] = sample[1]

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.data[idx],
            self.event[idx],
        )


def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]