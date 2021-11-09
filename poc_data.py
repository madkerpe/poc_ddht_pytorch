import numpy as np
import torch

length = 30
num_covariates = 4 # Don't change this
batch_size = 16


def generate_sample(id, event):
    # Generate a longitudional data sample, with 3 covariates,
    # Temporally 30 units long, no censoring yet
    # first dimention is just an id
    # second dimension a linear increasing value
    # third dimension is the sinus of the linear increasing value
    # TODO a fourth dimension that has all the information of the dying process
    # TODO add time to event to the dataloader
    # TODO left & right censoring

    # dataset in DDHT looks like this for each case: 
    #  (covariates, 
    #   mask vectors that indicates which covariates are missing,
    #   time-delta between each measurement
    #   time till event
    #   actual event)

    data = torch.zeros(length, num_covariates)
    data[:, 0] = id * torch.ones(length)

    random_start = length + 10 * torch.randn(1)

    if event == 0:
        data[:, 1] = random_start + torch.tensor(np.linspace(0, 2 * np.pi, num=length))
        data[:, 2] = torch.sin(data[:, 1])

    elif event == 1:
        data[:, 1] = random_start + torch.tensor(np.linspace(0, 2 * np.pi, num=length))
        data[:, 2] = 2 * torch.sin(data[:, 1])


    # we need a number between 1 and the time horizon to make the even happen
    data[:, 3] = np.exp((-1)*(data[:, 1] - random_start)) - ((random_start)/(2*(length+10)))

    return data


class PocDataset(torch.utils.data.Dataset):
    def __init__(self, num_cases=1024):
        torch.utils.data.Dataset.__init__(self)
        self.num_cases = num_cases
        self.labels = torch.randint(2, (num_cases,1))
        self.data = torch.zeros(num_cases, length, num_covariates)
        
        for i in range(num_cases):
            event = self.labels[i]
            self.data[i] = generate_sample(i, event)

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.data[idx], self.labels[idx])

test = generate_sample(3, 1)