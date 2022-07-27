import dask.dataframe as dd
import torch

min_age = 12
max_age = 180
epsilon = 1e-6

class FREDDIEMAC_main_dataset(torch.utils.data.Dataset):
    """
    test_set: does nothing at the moment
    """
    def __init__(self, dataframe, allowed_covariates, TIME_TO_EVENT_covariate, TOTAL_OBSERVED_LENGTH_covariate, LABEL_covariate, frac_cases=0.5, random_state=42, test_set=False, augment=False, data_augment_factor=3):
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
        self.TOTAL_OBSERVED_LENGTH_covariate = TOTAL_OBSERVED_LENGTH_covariate
        self.LABEL_covariate = LABEL_covariate

        self.loan_sequence_numbers = dataframe["LOAN_SEQUENCE_NUMBER"].unique().sample(frac=frac_cases, random_state=random_state).compute()
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

        print("len(lsn_series) = ", len(lsn_series))

        lsn_data = dd.merge(self.dataframe, lsn_series, on="LOAN_SEQUENCE_NUMBER", how="inner").compute()
        batch_length = len(lsn_series)

        print("batch_length = ", batch_length)

        lsn_label = lsn_data.groupby("LOAN_SEQUENCE_NUMBER").first()
        lsn_label = lsn_label[["LOAN_SEQUENCE_NUMBER", self.TIME_TO_EVENT_covariate, self.TOTAL_OBSERVED_LENGTH_covariate, self.LABEL_covariate]]

        #TODO don't iterate over the sequence, for now I'm doing exactly that,
        # what are you gonna do about it?
        data = torch.zeros((batch_length, self.max_length, self.num_covariates))
        for memory_index, value in enumerate(lsn_label.iterrows()):
            loan_entry = value[1]
            loan_length = loan_entry[self.TOTAL_OBSERVED_LENGTH_covariate]

            loan_data = lsn_data[lsn_data["LOAN_SEQUENCE_NUMBER"] == loan_entry["LOAN_SEQUENCE_NUMBER"]][self.allowed_covariates].to_numpy()
            loan_data = torch.tensor(loan_data)
            data[memory_index, :loan_length, :] = loan_data

        data_length = torch.tensor(lsn_label[self.TOTAL_OBSERVED_LENGTH_covariate].to_numpy()).unsqueeze(1)
        event = torch.tensor(lsn_label[self.LABEL_covariate].to_numpy()).unsqueeze(1)
        tte = torch.tensor(lsn_label[self.TIME_TO_EVENT_covariate].to_numpy()).unsqueeze(1)

        return (
            data,
            data_length,
            event,
            tte
        )



class FREDDIEMAC_main_dataloader():
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
