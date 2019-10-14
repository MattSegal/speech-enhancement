import torch
from torch import nn


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
        self.mse = nn.MSELoss()

    def __call__(self, input_audio, predicted_audio, target_audio):
        """
        Return single element loss tensor, containg loss value.
            predicted_audio is a tensor (batch_size, audio_length)
            target_audio is a tensor (batch_size, audio_length)
        """
        assert predicted_audio.shape == target_audio.shape
        assert len(predicted_audio.shape) == 2
        batch_size = predicted_audio.shape[0]
        predict_input = predicted_audio.view(batch_size, 1, -1)
        target_input = target_audio.view(batch_size, 1, -1)
        # Make predictions, get feature layers
        _ = self.loss_net(predict_input)
        pred_feature_layers = self.loss_net.feature_layers
        _ = self.loss_net(target_input)
        target_feature_layers = self.loss_net.feature_layers

        loss = torch.tensor([0.0], requires_grad=True).cuda()
        for idx in range(len(pred_feature_layers)):
            predicted_feature = pred_feature_layers[idx]
            target_feature = target_feature_layers[idx]
            loss += l1_loss(predicted_feature, target_feature)

        # Penalize model for not conserving power
        true_noise = input_audio - target_audio
        pred_noise = input_audio - predicted_audio
        noise_error = 10 * self.mse(true_noise, pred_noise)
        signal_error = 10 * self.mse(target_audio, predicted_audio)

        return loss + noise_error + signal_error


def mean_error_loss(predicted_t, target_t):
    return (predicted_t.mean().abs() - target_t.mean().abs()).abs()


def l1_loss(predicted_t, target_t):
    """
    Least absolute deviation / L1 loss function.
    Loss is defined as the sum of all the absolute  difference between the predicted and target values.

    Assume both input tensors have shape (batch_size, audio_length)
    """
    assert predicted_t.shape == target_t.shape
    assert len(predicted_t.shape) > 1
    diff_t = predicted_t - target_t
    return diff_t.abs().mean()
