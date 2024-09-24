from Iter import Iter
from Pairs import Pairs

class Puzzle(Iter):
    # We require J to be a dict with 'train' and 'test' keys
    def __init__(self, J):
        assert isinstance(J, dict)

        self.train = Pairs(J['train'])
        self.test = Pairs(J['test'])

        # Initialize the Iter base class with the list of pairs
        super().__init__([self.train, self.test])

    def overview(self):
        self.train.overview()
