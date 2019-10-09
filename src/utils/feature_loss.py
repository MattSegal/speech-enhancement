class AudioFeatureLoss:
    """
    Custom loss function, where loss is the distance
    between two feature vectors produced by the supplied loss network
    """

    def __init__(self, loss_net):
        """
        Store loss net for use in calculating feature vectors.
        Loss net must accept a tensor (batch_size, 1, audio_length)
        And expose property `feature_layers` - which returns a list of weight vectors.
        """
        self.loss_net = loss_net

    def __call__(self, predicted_audio, target_audio):
        """
        Return single element loss tensor, containg loss value.
            predicted_audio is a tensor (batch_size, audio_length)
            target_audio is a tensor (batch_size, audio_length)
        """
        # create "features" method on loss net?
        _ = loss_net(predicted_audio)
        pred_feature_layers = loss_net.feature_layers
        _ = loss_net(target_audio)
        target_feature_layers = loss_net.feature_layers

        loss = 0
        for idx in range(len(pred_feature_layers)):
            predicted_feature = pred_feature_layers[idx]
            target_feature = target_feature_layers[idx]
            loss += l1_loss(predicted_feature, target_feature)

        return torch.tensor([loss])


def least_abs_deviation_loss(predicted_t, current_t):
    """
    Least absolute deviation / L1 loss function.
    Loss is defined as the sum of all the absolute  difference between the predicted and target values.
    """
    diff_t = predicted_t - current_t
    return diff_t.abs().mean()
