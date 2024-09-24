from Iter import Iter
from Pair import Pair

class Pairs(Iter):
    # k is an iterable of "pairs", where each "pair" must be:
    # - either a dict with 'input' and 'output' keys
    # - or an iterable with 2 elements (input and output)
    # Either way, we get 2 values, which must be convertible to Grid objects
    # So list-of-lists, 2D np.array, Grid, grid-as-str, etc. all are ok.
    def __init__(self, k):
        def make_pair(u):
            return Pair(u['input'], u['output']) if isinstance(u, dict)  \
                    else Pair(*u)
        super().__init__(
            make_pair(u) for u in k
        )

    @property
    def X(self):
        return Iter(u.X for u in self)

    @property
    def Y(self):
        return Iter(u.Y for u in self)

    def overview(self):
        for pair in self:
            pair.overview()
            print()
