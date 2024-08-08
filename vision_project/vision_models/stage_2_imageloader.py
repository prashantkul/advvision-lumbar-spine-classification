from imageloader import ImageLoader

class Stage2ImageLoader(ImageLoader):

    def __init__(
        self, image_dir, label_coordinates_csv, labels_csv, roi_size, batch_size
    ):
        super(ImageLoader)

    def augment(self, dataset, predictions, ):
        predictions = predictions
        stage_2_labels = self.labels_csv
        df = pd.read_csv(self.label_coordinates_csv)
        df[['series_id','study_id']].drop_duplicates()

        ### need to match tensor of images and tensor of stage 1 predictions to list of unique study/series compound keys
        ### from there reshape each tensor, consolidating by study id

