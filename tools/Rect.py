from collections.abc import Iterable
from math import ceil, floor
from numbers import Number

from Coord import Coord

from types import GeneratorType
import numpy as np

class Rect:
    def __init__(self, *args, center=None, radius=None, **kwargs):
        self._t = self._b = self._l = self._r = 0

        if args:
            if len(args) == 1:
                a = args[0]
                a = tuple(a) if isinstance(a, GeneratorType)  \
                    else tuple(a.flatten) if isinstance(a, np.ndarray)  \
                        else a.tlbr if isinstance(a, Rect)  \
                            else (a.y, a.x) if isinstance(a, Coord)  \
                                else (a,) if isinstance(a, (Number, bool))  \
                                    else a if isinstance(a, (list, tuple))  \
                                        else 'wtf?!'

                assert a != 'wtf?!'

                while len(a) < 4:
                    a *= 2

                self.tlbr = [int(u) for u in a]

            elif len(args) == 2:
                # We need to be careful whether we're dealing with (t, l) or (tl, br)
                a, b = args
                is_number = lambda u: isinstance(u, (bool, Number))
                if is_number(a) and is_number(b):
                    self.tl = (a, b)
                else:
                    self.tl, self.br = Coord(a), Coord(b)

            elif len(args) == 4:
                self.tlbr = args

            else:
                assert False

        if center is not None:
            self.tl_br = (
                Coord(center) - radius,
                Coord(center) + radius
            )

        for k, v in kwargs.items():
            setattr(self, k, v)


    def __getattr__(self, k):
        # Fix issue with IPython _foo attributes
        # https://stackoverflow.com/questions/78899267/
        # Grid.__getattr__ does the same
        if k.startswith('_'):
            return

        is_property = isinstance(getattr(self.__class__, k, None), property)

        if is_property:
            return object.__getattribute__(self, k)

        if '_' in k:
            return [getattr(self, u) for u in k.split('_')]

        if len(k) > 1:
            is_lower = k.islower()
            k = k.lower()
            L = [getattr(self, u) for u in k]
            return Coord(L) if len(L) == 2 and is_lower else L

        assert False

    def __setattr__(self, k, v):
        if k.startswith('_'):
            object.__setattr__(self, k, v)
            return

        is_property = isinstance(getattr(self.__class__, k, None), property)

        if is_property:
            object.__setattr__(self, k, v)
            return

        # We need to check whether the attribute actually exists for the object, e.g. 'center' DOES exist as a property

        # This should only execute if there's NO corresponding attribute
        if len(k) == 1:
            object.__setattr__(self, k, v)
        else:
            if '_' in k:
                k = k.split('_')
            if not isinstance(v, Iterable):
                v = [v] * len(k)
            for k_, v_ in zip(k, v):
                setattr(self, k_, v_)


    def _make_property(attr):
        return property(
            lambda self: getattr(self, f'_{attr}'),
            lambda self, v: setattr(self, f'_{attr}', v)
        )
            # getter, setter)

    t = _make_property('t')
    b = _make_property('b')
    l = _make_property('l')
    r = _make_property('r')

    @property
    def h(self):
        return self.b - self.t + 1

    @h.setter
    def h(self, v):
        self.b = self.t + v - 1

    @property
    def w(self):
        return self.r - self.l + 1

    @w.setter
    def w(self, v):
        self.r = self.l + v - 1


    # Note `R.center.y += 1` will fail. That's as it should be.
    @property
    def center(self):
        return (self.tl + self.br) / 2

    # TODO: We're re-inventing the wheel in a couple of places
    # If we could bounce from __setattr__ to here, we could save some pain.
    @center.setter
    def center(self, v):
        self += -self.center + Coord(v)


    # Allow: newR = R(t=0, b=1)
    def __call__(self, **D):
        R = Rect(self.t, self.b, self.l, self.r)
        R.set(**D)
        return R

    # Allow R.set(t=0)
    def set(self, radius=None, **D):
        if radius is not None:
            self.tl_br = (
                self.center - radius,
                self.center + radius
            )
        for k, v in D.items():
            setattr(self, k, v)
        return self

    @property
    def copy(self):
        t, l, b, r = self.tlbr
        return Rect(t, l, b, r)

    def with_(self, **D):
        return self.copy.set(**D)


    @property
    def int(self):
        return Rect([int(u) for u in self.tlbr])

    def __repr__(self):
        return f'{self.tl:~>2} to {self.br:~>2}'

    def _slice(self, fix_right_margin=False):
        return tuple(
            slice(
                ceil(p),
                None if fix_right_margin and q == 0 else floor(q) + 1
            )
            for (p, q) in self.tb_lr
        )

    @property
    def slice(self):
        return self._slice()

    @property
    def mslice(self):
        return self._slice(fix_right_margin=True)


    def __contains__(self, p):
        t, l, b, r = self.tlbr
        y, x = Coord(p)
        return t <= y <= b and l <= x <= r

    def __iadd__(self, u):
        u = Coord(u)
        self.tl += u
        self.br += u
        return self

    def __add__(self, u):
        u = Coord(u)
        return Rect(tl=self.tl + u, br=self.br + u)

    def __sub__(self, u):
        u = Coord(u)
        return Rect(tl=self.tl - Coord(u), br=self.br - u)

    def __and__(self, u):
        if not isinstance(u, Rect):
            u = Rect(u)

        t, l, b, r = self.tlbr
        T, L, B, R = u.tlbr

        R = Rect(t=max(t, T), l=max(l, L), b=min(b, B), r=min(r, R))

        return None if R.r < R.l or R.b < R.t else R

    def __or__(self, u):
        if not isinstance(u, Rect):
            u = Rect(u)

        t, l, b, r = self.tlbr
        T, L, B, R = u.tlbr

        return Rect(t=min(t, T), l=min(l, L), b=max(b, B), r=max(r, R))

    def __eq__(self, u):
        if not isinstance(u, Rect):
            u = Rect(u)

        return Rect(p == q for p, q in zip(self.tlbr, u.tlbr))

    def __matmul__(self, u):
        if not isinstance(u, Rect):
            u = Rect(u)

        return all((self == u).tlbr)

    def __mul__(self, k):
        return Rect(k * u for u in self.tlbr)

    def extend(self, *args, in_place=False, λ=1, **kwargs):
        tl, br = Rect(*args, **kwargs).tl_br
        tl *= λ
        br *= λ

        if in_place:
            self.tl -= tl
            self.br += br
            return self
        else:
            return Rect(tl=self.tl - tl, br=self.br + br)

    def shrink(self, *args, **kwargs):
        return self.extend(*args, λ=-1, **kwargs)