import torch

# Only use first few layers for loss function.
LOSS_LAYERS = 6
CALCULATE_CALL_COUNT = 10


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
        self.reset_loss_tracking()

    def calculate_loss_weights(self):
        for idx in range(LOSS_LAYERS):
            layer_loss_avg = sum(
                [losses[idx] for losses in self.layer_loss_history]
            ) / float(CALCULATE_CALL_COUNT)
            self.layer_weights[idx] = layer_loss_avg

    def reset_loss_tracking(self):
        self.calls = 0
        self.layer_weights = [1.0 for _ in range(LOSS_LAYERS)]
        self.layer_loss_history = [[] for _ in range(CALCULATE_CALL_COUNT)]

    def __call__(self, predicted_audio, target_audio):
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

        # create "features" method on loss net?
        _ = self.loss_net(predict_input)
        pred_feature_layers = self.loss_net.feature_layers[:LOSS_LAYERS]
        _ = self.loss_net(target_input)
        target_feature_layers = self.loss_net.feature_layers[:LOSS_LAYERS]

        [z.size() for z in target_feature_layers]

        # TODO: Normalize each layer's loss by its running mean after some warm up period.
        # The weights are set to balance the contribution of each layer to the loss
        # Take average loss for each layer, over 1st 10 layers, then divide loss by that
        # For first 10 training epochs use a warm up value of 1

        weight = 1
        loss = torch.tensor([0.0], requires_grad=True).cuda()
        for idx in range(len(pred_feature_layers)):
            predicted_feature = pred_feature_layers[idx]
            target_feature = target_feature_layers[idx]
            raw_loss = l1_loss(predicted_feature, target_feature)
            loss[0] += (
                l1_loss(predicted_feature, target_feature) / self.layer_weights[idx]
            )
            if self.calls < CALCULATE_CALL_COUNT:
                self.layer_loss_history[self.calls].append(raw_loss)

        waveform_mean_error = (
            predicted_audio.mean().abs() - target_audio.mean().abs()
        )
        self.calls += 1
        if self.calls == CALCULATE_CALL_COUNT:
            self.calculate_loss_weights()

        return loss + waveform_mean_error.item()


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
