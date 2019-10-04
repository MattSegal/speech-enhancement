class AudioFeatureLoss:
    """
    Custom loss function, where loss is the distance
    between two feature vectors produced by the supplied loss network
    """

    def __init__(self, loss_net):
        """
        Store loss net for use in calculating feature vectors.
        Loss net must accept TODO
        And have interface TODO
        """
        self.loss_net = loss_net

    def __call__(self, predicted_audio, target_audio):
        """
        Return single element loss tensor, containg loss value.

        predicted_audio is a tensor (batch_size, )
        """
        # create "features" method on loss net?
        predicted_features = loss_net(predicted_audio)
        target_features = loss_net(target_audio)

        loss = 0
        for idx in range(6):
            predicted_feature = predicted_features[idx]
            target_feature = target_features[idx]
            loss += l1_loss(predicted_feature, target_feature)

        loss = sum(losses)
        return torch.tensor([loss])


def least_abs_deviation_loss(predicted_t, current_t):
    """
    Least absolute deviation / L1 loss function.
    Loss is defined as the sum of all the absolute  difference between the predicted and target values.
    """
    diff_t = predicted_t - current_t
    return diff_t.abs().mean()
