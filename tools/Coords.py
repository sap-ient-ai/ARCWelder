import numpy as np
from types import GeneratorType

from Iter import Iter
from Coord import Coord
from Rect import Rect

fix = lambda u: u.A if isinstance(u, Coords) else Coord(u)._

class Coords(Iter):
    def __init__(self, s):
        if isinstance(s, str):
            if isinstance(s, str):
                assert '\uFE0F' not in s, f'Coord: Variation Selector fail: {s}'

            s = s  \
                .replace('✳', '❇ ❎')  \
                .replace('❇', '▶ ◀ 🔼 🔽')  \
                .replace('❎', '↗ ↘ ↙ ↖')  \
                .split()

        if isinstance(s, GeneratorType):
            s = list(s)

        if isinstance(s, (list, tuple, Iter)):
            s = [Coord(u)._ for u in s]

        A = np.array([*np.ndindex(*s.hw)]) + s.tl._ if isinstance(s, Rect)  \
            else np.array(s)

        super().__init__(Coord(u) for u in A)

    def sort(self, *args, **kwargs):
        if len(args) > 0:
            a = args[-1]
            if isinstance(a, str) and a in '<>':
                kwargs['reverse'] = {'<': True, '>': False}[a]
                args = args[:-1]

        if len(args) > 0:
            kwargs['λ'] = args[0]

        λ = kwargs.get('λ', lambda u: u.mag2)
        reverse = kwargs.get('reverse', False)

        return Coords(super().sort(λ, reverse))

    @property
    def y(self):
        return Iter(u.y for u in self)

    @property
    def x(self):
        return Iter(u.x for u in self)

    @property
    def A(self):
        # TODO: We'd like to do np.array(self) but because we've overridden Iter.__getattr__, numpy searches for an attribute error "ValueError: invalid __array_struct__"
        # We should probably fix this in Iter
        return np.array([[u.y, u.x] for u in self])


    def __add__(self, u):
        return Coords(self.A + fix(u))

    def __neg__(self):
        return Coords(-self.A)

    def __sub__(self, u):
        return Coords(self.A - fix(u))

    def __mul__(self, u):
        return Coords(self.A * fix(u))


    def __repr__(self):
        return ', '.join(map(str, self))


    @property
    def bounds(self):
        Y, X = self.y, self.x
        return Rect(tlbr=[min(Y), min(X), max(Y), max(X)])
