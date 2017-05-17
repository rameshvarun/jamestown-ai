class EWMA:
    """Keep track of an exponentially weighted moving average. Used for FPS."""

    def __init__(self, alpha, initial=0):
        self.value = initial
        self.alpha = alpha

    def observe(self, value):
        self.value = self.alpha * value + (1 - self.alpha) * self.value

    def get(self):
        return self.value