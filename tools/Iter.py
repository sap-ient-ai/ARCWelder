from types import FunctionType, LambdaType

from itertools import permutations

from inspect import isclass

def is_func(u):
    return isinstance(u, (FunctionType, LambdaType))

class Iter:
    def __init__(self, iterable):
        self.L = list(iterable)
        self.reset()


    def reset(self):
        self._iter = iter(self.L)

    def __iter__(self):
        return iter(self.L)

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return repr(self.L)

    def __len__(self):
        return len(self.L)

    def __getitem__(self, key):
        y = [self[u] for u in key] if isinstance(key, list)  \
            else self.L[key]
        return type(self)(y) if isinstance(key, (slice, list))  \
            else y

    @property
    def flatten(self):
        return Iter( subitem
            for item in self.L
                for subitem in (
                    item.flatten if isinstance(item, Iter) else [item]
                )
        )

    @property
    def permutations(self):
        return Iter(permutations(self.L))

    @property
    def i(self):
        return iter(self.L)

    @property
    def Σ(self):
        return len(self.L)

    def __rmatmul__(self, u):  # f @ Iter
        return Iter(u(w) for w in self)  if is_func(u) or isclass(u) \
            else super().__rmatmul__(u)

    # 2 uses:
    # - `items | cond` (items where condition is True)
    # - `iterA | iterB` (concat)
    def __or__(self, u):
        return type(self)(v for v in self if u(v))  if is_func(u) \
            else type(self)(self.L + u.L) if isinstance(u, Iter)  \
                else super().__or__(u)

    def sort(self, λ, reverse=False):
        s = sorted(self, key=λ, reverse=reverse)
        return type(self)(s)

    # TODO: Fix up Set to handle this?
    @property
    def unique(self):
        D = {str(u): u for u in self}
        return Iter(D.values())

    @property
    def counts(self):
        count = lambda u: len([v for v in self if str(v) == str(u)])
        return Iter((u, count(u)) for u in self.unique)

    @property
    def cycler(self):
        while True:
            for item in self.L:
                yield item

    def pairs(self, same=True):
        L = self.L
        N = len(L)
        ok = lambda i, j: True if same else i != j
        return Iter((L[i], L[j]) for i in range(N) for j in range(N) if ok(i, j))

    def __getattr__(self, k):
        # print('ITER GETATTR:' + k)
        # Fix issue with IPython _foo attributes
        # https://stackoverflow.com/questions/78899267/
        # Grid.__getattr__ does the same
        if k.startswith('_'):
            return

        if (is_property := isinstance(getattr(self.__class__, k, None), property)):
            return object.__getattribute__(self, k)

        return Iter(getattr(u, k) for u in self)

class Cycler(Iter):
    def __init__(self, k):
        if isinstance(k, str):
            k = k.split()
        super().__init__(k)

    def __iter__(self):
        while True:
            super().reset()
            yield from super().__iter__()
