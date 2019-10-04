class MovingAverage:
    """
    Tracks moving average of loss function using exponential decay.
    """

    def __init__(self, decay):
        self.value = 0
        self.decay = decay

    def update(self, value):
        self.value = self.decay * self.value + (1 - self.decay) * value
