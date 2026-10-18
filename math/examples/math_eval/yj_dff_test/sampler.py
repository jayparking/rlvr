"""TRL-compatible repetition without reshuffling the prepared schedule."""


class OrderedGroupSampler:
    def __init__(self, size, generations):
        self.size = size
        self.generations = generations

    def __iter__(self):
        for index in range(self.size):
            for _ in range(self.generations):
                yield index

    def __len__(self):
        return self.size * self.generations
