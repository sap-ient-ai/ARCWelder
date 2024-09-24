import numpy as np

import numbers

def convert(u):
    return int(u) if isinstance(u, np.int64)  \
        else float(u) if isinstance(u, np.float64)  \
            else u

class Coord:
    # 🦋 init

    # allow Coord(1), Coord(2, 3), Coord((4, 5)), Coord(np.array[6])
    def __init__(self, *U):
        assert len(U) in [1, 2]

        V = U if len(U) == 2 else U[0]

        if isinstance(V, str):
            assert '\uFE0F' not in V, f'Coord: Variation Selector fail: {V}'

            self._ = np.array({
                '⏺': (0, 0),

                '▶': (0, 1),
                '◀': (0, -1),
                '🔼': (-1, 0),
                '🔽': (1, 0),

                '↗': (-1, 1),
                '↖': (-1, -1),
                '↙': (1, -1),
                '↘': (1, 1),
            }[V])

        elif isinstance(V, Coord):
            self._ = V._.copy()

        elif isinstance(V, (list, tuple)):
            self._ = np.array(V)

        elif isinstance(V, np.ndarray):
            self._ = V

        else:
            assert isinstance(V, numbers.Number)
            self._ = np.ones((2)).astype(int) * V


    # 🦋 operators

    def __add__(self, other):
        return Coord(self._ + Coord(other)._)

    def __sub__(self, other):
        return Coord(self._ - Coord(other)._)

    def __mul__(self, other):
        return Coord(self._ * Coord(other)._)

    def __neg__(self):
        return Coord(-self._)

    def __truediv__(self, other):
        return Coord(self._ / Coord(other)._)

    def __floordiv__(self, other):
        return Coord(self._ // Coord(other)._)

    def __eq__(self, other):
        return (self._ == Coord(other)._).all()

    # 🦋 properties

    @property
    def y(self):
        return convert(self._[0])

    @property
    def x(self):
        return convert(self._[1])

    @property
    def h(self):
        return self.y

    @property
    def w(self):
        return self.x

    @property
    def abs(self):
        return Coord(np.abs(self._))

    @property
    def mag2(self):
        return self.x**2 + self.y**2

    def dist2(self, p):
        return (self - p).mag2

    def T(self):
        return Coord(self.y, self.x)

    @property
    def min(self):
        return min(self.y, self.x)

    @property
    def max(self):
        return max(self.y, self.x)


    # 🦋 printing

    def __repr__(self):
        return f'{self:~>2}'  # invoke __format__

    # TODO: handle -13.5, currently it'll be too close. Need ':~±xx.x'
    def __format__(self, spec):
        orig, *custom = spec.split('~')
        custom = custom[0] if custom else '>2'
        y, x = convert(self.y), convert(self.x)
        xx_x = lambda u: f'{int(u):>2}. ' if u == int(u) else f'{u:>4.1f}'
        u = {
            'x': f'{y}x{x}',
            'x5': f'{y:>2}x{x:<2}',
            '>2': f'{y:>2} {x:>2}',
            'xx.x': f'{xx_x(y)} {xx_x(x)}',
        }.get(custom, f'{y} {x}')
        return format(u, orig)

    def __iter__(self):
        yield from (self.y, self.x)

    def independent(self, u):
        p, q = self._, Coord(u)._
        _3d = lambda u: np.append(u, 0)
        return np.cross(_3d(p), _3d(q))[2] != 0

    def ortho(self, u):
        p, q = self._, Coord(u)._
        return np.dot(p, q) == 0

    def colinear(self, u):
        return not self.independent(u)

    @property
    def glyph(self):
        return next(
            (glyph for glyph in '⏺▶◀🔼🔽↗↖↙↘'
                if self == Coord(glyph)),
            '❌'
        )

    @property
    def int(self):
        return Coord(int(self.y), int(self.x))

    @property
    def A(self):
        return self._
