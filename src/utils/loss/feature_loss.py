import torch
from torch import nn

from .l1_loss import l1_loss


class AudioFeatureLoss:
    """
    Custom loss function, where loss is the distance
    between two feature vectors produced by the supplied loss network
    """

    def __init__(self, loss_net, use_cuda=True):
        """
        Store loss net for use in calculating feature vectors.
        Loss net must accept a tensor (batch_size, 1, audio_length)
        And expose property `feature_layers` - which returns a list of weight vectors.
        """
        self.loss_net = loss_net
        self.use_cuda = use_cuda

    def get_feature_loss(self, predicted_audio, target_audio):
        assert predicted_audio.shape == target_audio.shape
        batch_size = predicted_audio.shape[0]
        predict_input = predicted_audio.view(batch_size, 1, -1)
        target_input = target_audio.view(batch_size, 1, -1)

        # Make predictions, get feature layers.
        self.loss_net(predict_input)
        pred_feature_layers = self.loss_net.feature_layers

        self.loss_net(target_input)
        target_feature_layers = self.loss_net.feature_layers

        # Sum up l1 losses over all feature layers.
        loss = torch.tensor([0.0], requires_grad=True)
        loss = loss.cuda() if self.use_cuda else loss
        for idx in range(len(pred_feature_layers)):
            predicted_feature = pred_feature_layers[idx]
            target_feature = target_feature_layers[idx]
            loss = loss + l1_loss(predicted_feature, target_feature)

        return loss

    def __call__(self, input_audio, predicted_audio, target_audio):
        """
        Return single element loss tensor, containg loss value.
            predicted_audio is a tensor (batch_size, audio_length)
            target_audio is a tensor (batch_size, audio_length)
        """
        # Calculate noise.
        true_noise = input_audio - target_audio
        pred_noise = input_audio - predicted_audio

        # Get feature losses for clean audio and noise
        clean_feature_loss = self.get_feature_loss(predicted_audio, target_audio)
        noise_feature_loss = self.get_feature_loss(pred_noise, true_noise)
        return clean_feature_loss + noise_feature_loss
