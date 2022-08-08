import dask.dataframe as dd
import pandas as pd
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

        self.df_loan_sequence_data = self.compute_loan_sequence_numbers(dataframe, augment, data_augment_factor)
        self.num_cases = len(self.df_loan_sequence_data)
        self.num_covariates = len(self.dataframe[allowed_covariates].columns)

    def compute_loan_sequence_numbers(self, dataframe, augment, augment_factor):
        dd_lsn_data = dataframe[["LOAN_SEQUENCE_NUMBER", self.TOTAL_OBSERVED_LENGTH_covariate, self.TIME_TO_EVENT_covariate, self.LABEL_covariate]]
        df_lsn_data = dd_lsn_data.drop_duplicates().compute()
        
        if augment:
            original_df_lsn_data = df_lsn_data.copy() # create a copy

            for augment_iteration in range(augment_factor):
                augmented_lsn_data = original_df_lsn_data.copy() # create a copy

                def augment(x):
                    loan_length = round(x[self.TOTAL_OBSERVED_LENGTH_covariate])
                    loan_label = x[self.LABEL_covariate]

                    if loan_length > 2*self.min_length and (loan_label == 0 or loan_label == 1 or loan_label == 2):
                        x[self.TOTAL_OBSERVED_LENGTH_covariate] = int(torch.randint(min_age, loan_length + 1,(1,)))
                    return x
                        
                augmented_lsn_data = augmented_lsn_data.apply(lambda x: augment(x), axis=1)
                df_lsn_data = pd.concat([df_lsn_data, augmented_lsn_data])

        df_lsn_data = df_lsn_data.sample(frac=1).reset_index(drop=True)
        self.num_cases = len(df_lsn_data)

        return df_lsn_data
    
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

        lsn_labels = self.df_loan_sequence_data.iloc[idx]
        lsn_data = dd.merge(self.dataframe, lsn_labels["LOAN_SEQUENCE_NUMBER"], on="LOAN_SEQUENCE_NUMBER", how="inner").compute()
        batch_length = len(lsn_labels)

        #TODO don't iterate over the sequence, for now I'm doing exactly that,
        # what are you gonna do about it?
        data = torch.zeros((batch_length, self.max_length, self.num_covariates))
        for memory_index, value in enumerate(lsn_labels.iterrows()):
            loan_entry = value[1]
            loan_length = round(loan_entry[self.TOTAL_OBSERVED_LENGTH_covariate])

            dd_loan_data = lsn_data[lsn_data["LOAN_SEQUENCE_NUMBER"] == loan_entry["LOAN_SEQUENCE_NUMBER"]][self.allowed_covariates]
            loan_data = dd_loan_data.to_numpy()
            loan_data = torch.tensor(loan_data)

            data[memory_index, :loan_length, :] = loan_data[:loan_length, :]

        event = torch.tensor(lsn_labels[self.LABEL_covariate].to_numpy()).unsqueeze(1)
        data_length = torch.tensor(lsn_labels[self.TOTAL_OBSERVED_LENGTH_covariate].to_numpy()).type(torch.int64).unsqueeze(1)
        tte = torch.tensor(lsn_labels[self.TIME_TO_EVENT_covariate].to_numpy()).type(torch.int64).unsqueeze(1)

        return (
            data,
            data_length,
            event,
            tte
        )


"""
        lsn_series =  self.df_loan_sequence_data.iloc[idx]

        lsn_data = dd.merge(self.dataframe, lsn_series, on="LOAN_SEQUENCE_NUMBER", how="inner").compute()
        batch_length = len(lsn_series)

        lsn_data["LOAN_SEQUENCE_NUMBER_GROUP_INDEX"] = lsn_data["LOAN_SEQUENCE_NUMBER"].astype(str)
        lsn_label = lsn_data.groupby("LOAN_SEQUENCE_NUMBER_GROUP_INDEX").first()


"""

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
