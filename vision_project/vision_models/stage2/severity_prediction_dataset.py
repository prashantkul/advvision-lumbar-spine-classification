import pandas as pd

from vision_project.vision_models import constants
from vision_project.vision_models.dataset import Dataset


class SeverityPredictionDataset(Dataset):

    def __init(self, disease_predictions):
        super().__init__(constants.BATCH_SIZE)
        self.disease_predictions = disease_predictions

    def augment(self):
        self.train_df = \
            pd.merge(self.train_df, self.disease_predictions, on=['series_id', 'study_id'], how='left')

        self.val_df = \
            pd.merge(self.val_df, self.disease_predictions, on=['series_id', 'study_id'], how='left')

        self.test_df = \
            pd.merge(self.test_df, self.disease_predictions, on=['series_id', 'study_id'], how='left')

