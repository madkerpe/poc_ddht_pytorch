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
        self.min_length = min_age
        self.max_length = max_age

        self.dataframe = dataframe

        self.allowed_covariates = allowed_covariates
        self.TIME_TO_EVENT_covariate = TIME_TO_EVENT_covariate
        self.LABEL_covariate = LABEL_covariate

        self.loan_sequence_numbers = dataframe["LOAN_SEQUENCE_NUMBER"].sample(frac=frac_cases, random_state=random_state).compute()
        self.num_cases = len(self.loan_sequence_numbers)
        self.num_covariates = len(self.dataframe[allowed_covariates].columns)

    def __len__(self):
        return self.num_cases

    def get_num_covariates(self):
        return self.num_covariates

    def get_min_length(self):
        return self.min_length

    def get_max_length(self):
        return self.max_length

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

    def __len__(self):
        return self.max_iterations

    def get_max_iterations(self):
        return self.max_iterations

    def __next__(self):
        if self.iterator_count >= self.max_iterations:
            self.iterator_count = 0
        
        sample = self.dataset[self.iterator_count * self.batch_size:(self.iterator_count + 1) * self.batch_size]
        return sample
