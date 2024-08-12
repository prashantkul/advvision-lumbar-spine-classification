import pandas as pd
from sklearn.model_selection import train_test_split
from vision_project.vision_models import constants
from vision_project.vision_models.dataset import Dataset


class SeverityPredictionDataset(Dataset):

    def __init(self, disease_predictions, balance_variables, ground_truth_predictions=None):
        super().__init__(constants.BATCH_SIZE)
        self.disease_predictions = disease_predictions
    ##TODO: we are predicting at the study_id level but we split by the series id

    def _create_split(self, df, balance_variables=None):
        # create a composite key of disease type, severity levels,  and study_id
        df['composite_key'] = '_'.join(str(var) for var in balance_variables)
        class_distribution = df['composite_key'].value_counts()

        train_ids, test_ids = train_test_split(
            class_distribution.index,
            test_size=0.2,
            stratify=class_distribution.values,
            random_state=42
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.25,
            stratify=class_distribution[train_ids].values,
            random_state=42
        )

        # Create splits
        train_split = df[df['composite_key'].isin(train_ids)].copy()
        val_split = df[df['composite_key'].isin(val_ids)].copy()
        test_split = df[df['composite_key'].isin(test_ids)].copy()

        return train_split, val_split, test_split, df

    def _prepare_data(self):
        # Read the label coordinates CSV file and create a DataFrame
        df = self._create_train_label_cord_dataframe()
        self.train_df, self.val_df, self.test_df, self.split_data = self._create_split(df, self.balance_variables)

        # Extract unique labels and store them from labels.csv
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()
        print("#"* 100)
        print("Dataset splits sizes:", self.get_df_sizes())

    def augment(self, predictions):
        self.train_df = \
            pd.merge(self.train_df, predictions, on=['study_id'], how='left')

        self.val_df = \
            pd.merge(self.val_df, predictions, on=['study_id'], how='left')

        self.test_df = \
            pd.merge(self.test_df, predictions, on=['study_id'], how='left')