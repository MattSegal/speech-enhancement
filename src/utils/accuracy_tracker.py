class AccuracyTracker:
    """
    Tracks average of prediciton accuracy.
    """

    def __init__(self, num_total):
        self.num_success = 0
        self.num_total = num_total

    def update(self, predictions, labels):
        """
        Update accuracy average.
        Predictions is a tensor (batch_size, num_classes) with a prediction for each class
        Labels is a tensor (batch_size,) containing the correct class label indexes
        """
        predicted_labels = predictions.argmax(dim=1)
        successful_predictions = predicted_labels == labels
        self.num_success += successful_predictions.sum().item()

    @property
    def accuracy(self):
        return self.num_success / float(self.num_total)
