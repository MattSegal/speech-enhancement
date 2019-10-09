class CombinedDataLoader:
    """
    Combines two data loaders, alternating between the two
    """

    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    @property
    def is_loader_a(self):
        return self.count % 2 == 0

    def __len__(self):
        return min([2 * len(self.loader_a), 2 * len(self.loader_b)])

    def __iter__(self):
        self.count = -1
        self.iter_a = self.loader_a.__iter__()
        self.iter_b = self.loader_b.__iter__()
        return self

    def __next__(self):
        self.count += 1
        return next(self.iter_a) if self.is_loader_a else next(self.iter_b)
