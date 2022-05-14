import numpy as np
import torch
import math
import matplotlib.pyplot as plt

min_age = 5
max_age = 12*3
num_covariates = 5
epsilon = 1e-6

def generate_sample():
    """
    info:
        length = amount of data of the full age that is captured, so age >= length

    latent data:
        X ~ Unif(0,1)
        Y ~ Unif(0,1)
        Z1 ~ Unif(0,1)
        Z2 ~ Unif(0,1)
        age = full time of the underlying process

    data:
        data[:, 0] --> current age in months
        data[:, 1] --> original duration in months
        data[:, 2] --> remaining months
        data[:, 3] --> X*sin(age + X)
        data[:, 4] --> Y*cos(age + Y)

    event:
        0 --> full repay
        1 --> prepay
        2 --> default
        3 --> censored

    censoring:
        if Z1 > 0.5 --> right censoring on position Z2*len(data)
        if Z1 < 0.5 --> no censoring
    """

    # define the latent variables X and Y
    X = torch.rand(1)
    Y = torch.rand(1)

    # deduce the event based on X and the age based on Y
    if X > 0.5:
        ground_truth_event = 0
        age = max_age

    elif X <= 0.25:
        ground_truth_event = 1
        age = max(math.floor(Y*max_age - epsilon), min_age)
        # length at least min_length, length max the max_length minus one

    else:
        ground_truth_event = 2
        age = max(math.floor(Y*max_age - epsilon), min_age)

    # reserve space for a data sample
    data = torch.zeros(age, num_covariates)

    # add some random noise on the start of the sample
    data[:, 0] = torch.arange(0, age, dtype=torch.float32)
    data[:, 1] = age*torch.ones(age)
    data[:, 2] = age - torch.arange(0, age, dtype=torch.float32) - 1
    data[:, 3] = X*torch.sin(torch.arange(0, age, dtype=torch.float32) + X)
    data[:, 4] = Y*torch.cos(torch.arange(0, age, dtype=torch.float32) + Y)

    # define the latent variables Z1 and Z2
    Z1 = torch.rand(1)
    Z2 = torch.rand(1)

    event = ground_truth_event

    # censoring based on Z1 and Z2
    if Z1 > 0.5 and math.floor(Z2*age) > min_age:
        # we do right censoring
        event = 3
        right_censor = math.floor(Z2*age)
        data = data[:right_censor]

    length = data.shape[0]

    meta = {'age': age, "ground_truth_event": ground_truth_event, 'X': X, 'Y': Y, 'Z1': Z1, 'Z2': Z2}

    return data, length, event, meta

def display_sample(sample_data, sample_length, sample_event):
    """
    This function displays the following dimensions from a single sample out of the PocDataset
    using matplotlib:
        data[:, 3] --> X*sin(age + X)
        data[:, 4] --> Y*cos(age + Y)
    """
    sample_data = sample_data.to(device='cpu')
    sample_length = sample_length.to(device='cpu')
    sample_event = sample_event.to(device='cpu')

    print("Length is %d, Event is %d" % (sample_length, sample_event))
    plt.plot(np.array(sample_data[:sample_length,0]), np.array(sample_data[:sample_length,3]))
    plt.plot(np.array(sample_data[:sample_length,0]), np.array(sample_data[:sample_length,4]))

class PocDataset(torch.utils.data.Dataset):
    """
    this dataset uses the generate_sample() function
    it padds the samples with zeros at the end and 
    puts them into a pytorch dataset class
    """
    def __init__(self, num_cases=1024):
        torch.utils.data.Dataset.__init__(self)
        self.num_cases = num_cases

        self.data = torch.zeros(num_cases, max_age, num_covariates)
        self.data_length = torch.zeros(num_cases, 1, dtype=torch.long)
        self.event = torch.zeros(num_cases, 1, dtype=torch.long)

        for i in range(num_cases):
            sample = generate_sample()
            sample_length = sample[1]
            self.data[i,:sample_length] = sample[0]
            self.data_length[i] = sample_length
            self.event[i] = sample[2]

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.data[idx],
            self.data_length[idx],
            self.event[idx]
        )