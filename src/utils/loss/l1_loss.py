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
