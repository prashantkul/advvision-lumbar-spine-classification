import pandas as pd
from sklearn.model_selection import train_test_split
from vision_models import constants
from vision_models.dataset import Dataset
from vision_models.stage2.helper import augment, tensorize


class SeverityPredictionDataset(Dataset):

    def __init__(self, disease_predictions, balance_variables):
        self.disease_predictions = disease_predictions
        self.balance_variables = balance_variables
        super().__init__(constants.BATCH_SIZE)

    def _create_split(self, df):
        # Create a composite key of disease type, severity levels, and study_id
        balance_vars = self.balance_variables
    
        # Create a composite key of the specified variables
        df['composite_key'] = df.apply(lambda row: '_'.join([str(row[var]) for var in balance_vars if var in df.columns]), axis=1)
    

        # Assign a composite key to each series_id
        series_key_mapping = df.groupby('series_id')['composite_key'].agg(lambda x: '_'.join(set(x))).reset_index()

        # Split the series_ids based on the composite key
        train_ids, test_ids = train_test_split(
            series_key_mapping['series_id'],
            test_size=0.25,
            stratify=series_key_mapping['composite_key'],
            random_state=42
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.25,
            stratify=series_key_mapping[series_key_mapping['series_id'].isin(train_ids)]['composite_key'],
            random_state=42
        )

        # Create splits by selecting rows that belong to the respective series_ids
        train_split = df[df['series_id'].isin(train_ids)].copy()
        val_split = df[df['series_id'].isin(val_ids)].copy()
        test_split = df[df['series_id'].isin(test_ids)].copy()

        return train_split, val_split, test_split, df

    def _prepare_data(self):
        # Read the label coordinates CSV file and create a DataFrame
        df = self._create_train_label_cord_dataframe()
        self.train_df, self.val_df, self.test_df, self.split_data = self._create_split(df)
        for df in [self.train_df, self.val_df, self.test_df, self.split_data]:
            tensorize(df, self.disease_predictions, label_columns=self.balance_variables)
        # self.train_df, self.val_df, self.test_df, self.split_data = tensorize(
        #     self._create_split(df), self.disease_predictions, label_columns=self.balance_variables)

        # Extract unique labels and store them from labels.csv
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()
        print("#"* 100)
        print("Dataset splits sizes:", self.get_df_sizes())
