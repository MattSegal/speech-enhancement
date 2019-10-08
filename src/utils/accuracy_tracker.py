class AccuracyTracker:
    """
    Calculates prediciton accuracy for an epoch of data.
    Useful for multi-class classification.
    """

    def __init__(self, num_total):
        self.num_success = 0
        self.num_total = num_total

    def update(self, predictions, labels):
        """
        Update number of successful predictions.
        Predictions is a tensor (batch_size, num_classes) with a prediction for each class
        Labels is a tensor (batch_size,) containing the correct class label indexes
        """
        predicted_labels = predictions.argmax(dim=1)
        successful_predictions = predicted_labels == labels
        self.num_success += successful_predictions.sum().item()

    @property
    def value(self):
        """
        0 is worst - none of the samples were correctly classified
        1 is best - all of of the samples were correctly classified
        """
        return self.num_success / float(self.num_total)


class HammingLossTracker:
    """
    Calculates Hamming loss for an epoch of data.
    Useful for multi-label classification.

    Reports the proportion of labels that are wrong.
    """

    def __init__(self, num_samples, num_classes):
        self.num_wrong = 0
        self.num_total = num_samples * num_classes

    def update(self, predictions, labels):
        """
        Update number of wrong predictions.
        Predictions is a tensor (batch_size, num_classes) with a prediction for each class
        Labels is a tensor (batch_size, num_classes) containing a 1 for present and 0 for absent
        """
        # Convert probabilities into predictions
        preds = predictions.clone()
        threshold = 0.5
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0
        failed_predictions = preds != labels
        self.num_wrong += failed_predictions.sum().item()

    @property
    def value(self):
        """
        0 is best - every class, for every label was correct
        1 is worst - every class, for every label was wrong
        """
        return self.num_wrong / float(self.num_total)
