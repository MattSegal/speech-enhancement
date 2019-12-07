from src.datasets.s3dataset import S3BackedDataset

DATASET_NAME = "speech_evaluation"


class SpeechEvaluationDataset(S3BackedDataset):
    def __init__(self, quiet=True):
        super().__init__(dataset_name=DATASET_NAME, quiet=quiet)
