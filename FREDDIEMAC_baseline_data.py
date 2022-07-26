import dask.dataframe as dd
import torch

min_age = 12
max_age = 180
epsilon = 1e-6

class FREDDIEMAC_basline_dataset(torch.utils.data.Dataset):
    """
    this dataset uses the generate_sample() function
    it padds the samples with zeros at the end and 
    puts them into a pytorch dataset class

    test_set: does nothing at the moment
    """
    def __init__(self, dataframe, allowed_covariates, TIME_TO_EVENT_covariate, LABEL_covariate, frac_cases=0.5, random_state=42, test_set=False, augment=False, data_augment_factor=3):
        torch.utils.data.Dataset.__init__(self)
        self.frac_cases = frac_cases
        self.test_set = test_set
        self.augment = augment
        self.augment_factor = data_augment_factor

        self.dataframe = dataframe #.compute()

        self.allowed_covariates = allowed_covariates
        self.TIME_TO_EVENT_covariate = TIME_TO_EVENT_covariate
        self.LABEL_covariate = LABEL_covariate

        self.loan_sequence_numbers = dataframe["LOAN_SEQUENCE_NUMBER"].sample(frac=frac_cases, random_state=random_state).compute()
        self.num_cases = len(self.loan_sequence_numbers)
        self.num_covariates = len(self.dataframe[allowed_covariates].columns)

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            idx = [idx]            

        lsn_series = self.loan_sequence_numbers.iloc[idx]
        lsn_data = dd.merge(self.dataframe, lsn_series, on="LOAN_SEQUENCE_NUMBER", how="inner")

        data = torch.tensor(lsn_data[self.allowed_covariates].compute().to_numpy()).unsqueeze(1)
        data_length = torch.tensor(lsn_data[self.TIME_TO_EVENT_covariate].compute().to_numpy()).unsqueeze(1)
        event = torch.tensor(lsn_data[self.LABEL_covariate].compute().to_numpy()).unsqueeze(1)
        tte = torch.tensor(lsn_data[self.TIME_TO_EVENT_covariate].compute().to_numpy()).unsqueeze(1)

        return (
            data,
            data_length,
            event,
            tte
        )

class FREDDIEMAC_dataloader():
    def __init__(self, dataset, batch_size):
        self.iterator_count = 0
        self.max_length = len(dataset)
        self.max_iterations = self.max_length // batch_size
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator_count >= self.max_iterations:
            raise StopIteration
        else:
            self.iterator_count += 1

            return self.dataset[self.iterator_count * self.batch_size:(self.iterator_count + 1) * self.batch_size]

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

    
