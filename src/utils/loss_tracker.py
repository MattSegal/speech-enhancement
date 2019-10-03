class LossTracker:
    """
    Tracks moving average of loss function using exponential decay.
    """

    def __init__(self, decay):
        self.loss = 0
        self.decay = decay

    def update(self, loss_value):
        self.loss = self.decay * self.loss + (1 - self.decay) * loss_value
