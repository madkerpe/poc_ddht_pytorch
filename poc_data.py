import numpy as np
import torch

length = 30
num_covariates = 3
batch_size = 16


def generate_sample(id, event):
    # Generate a longitudional data sample, with 3 covariates,
    # Temporally 30 units long, no censoring yet, TODO left or right censored
    # first dimention is just an id
    # second dimension a linear increasing value
    # third dimension is the sinus of the linear increasing value
    # TODO a fourth dimension that has all the information of the dying process  

    data = torch.zeros(length, num_covariates)
    data[:, 0] = id * torch.ones(length)

    random_start = 10 + 10 * torch.randn(1)

    if event == 0:
        data[:, 1] = random_start + torch.tensor(np.linspace(0, 2 * np.pi, num=length))
        data[:, 2] = torch.sin(data[:, 1])

    elif event == 1:
        data[:, 1] = random_start + torch.tensor(np.linspace(0, 2 * np.pi, num=length))
        data[:, 2] = 2 * torch.sin(data[:, 1])

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