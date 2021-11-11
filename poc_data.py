import numpy as np
import torch
import math

length = 30
num_covariates = 3  # This is not customisable


def generate_sample(id, latent_variable=None):
    """
    Generate a longitudional data sample, with 3 covariates,
    first dimention is just an id
    second dimension a linear increasing value
    third dimension is the sinus of the linear increasing value
    TODO maybe a fourth dimension that has more information of the dying process
    TODO left & right censoring

    There is one latent variable, a value between 0 and 1, the survival time is dependent 
    on it AND it is present in the longitudional data by the amplitude of the sine function

    dataset in DDHT looks like this for each case:
     (covariates, --> 3 covariates
      mask vectors that indicates which covariates are missing, --> not present here yet
      time-delta between each measurement --> here it's constant
      time till event --> present
      actual event --> present)
    """

    # define the latent variable
    if latent_variable == None:
        latent_variable = torch.rand(1)

    # reserve space for the data sample
    data = torch.zeros(length, num_covariates)
    data[:, 0] = id * torch.ones(length)

    # make the time horizon dependent on the latent variable
    time_to_event = 1 + math.floor((length - 1) * latent_variable)

    # add some random noise on the start of the sample
    random_start = 10 + 10 * torch.rand(1)
    data[:, 1] = random_start + torch.tensor(np.linspace(0, 5 * np.pi, num=length))

    # make the event dependent on the latent variable
    if latent_variable <= 0.5:
        event = 0

    if latent_variable > 0.5:
        event = 1
        data[:, 2] = 2 * torch.sin(data[:, 1])

    # make the longitudional data dependent on the latent variable
    data[:, 2] = latent_variable * torch.sin(data[:, 1])

    # set everything after the event to zero
    if time_to_event < length:
        data[time_to_event:, :] = torch.zeros(length - time_to_event, num_covariates)

    return data, time_to_event, event, latent_variable


class PocDataset(torch.utils.data.Dataset):
    def __init__(self, num_cases=1024):
        torch.utils.data.Dataset.__init__(self)
        self.num_cases = num_cases

        self.data = torch.zeros(num_cases, length, num_covariates)
        self.event = torch.zeros(num_cases, 1)
        self.time_to_event = torch.zeros(num_cases, 1)
        self._latent_variable = torch.zeros(num_cases, 1)

        for i in range(num_cases):
            data, time_to_event, event, _latent_variable = generate_sample(i)
            self.data[i] = data
            self.event[i] = event
            self.time_to_event[i] = time_to_event
            self._latent_variable[i] = _latent_variable

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.data[idx],
            self.event[idx],
            self.time_to_event[idx],
            self._latent_variable[idx],
        )