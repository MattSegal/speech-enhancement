class CombinedDataLoader:
    """
    Combines two data loaders, alternating between the two
    """

    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.iter_a = loader_a.__iter__()
        self.iter_b = loader_b.__iter__()
        self.a_finished = False
        self.b_finished = False
        self.count = -1

    @property
    def is_loader_a(self):
        return self.count % 2 == 0

    def __len__(self):
        return len(self.loader_a) + len(self.loader_b)

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.is_loader_a and not self.a_finished:
            try:
                sample = next(self.iter_a)
                return sample
            except StopIteration:
                self.a_finished = False
        else:
            try:
                sample = next(self.iter_b)
                return sample
            except StopIteration:
                self.b_finished = False

        if self.a_finished and self.b_finished:
            raise StopIteration()

