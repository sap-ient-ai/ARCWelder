from collections import OrderedDict

class Set:
    def __init__(self, u=None):
        if u is None:
            u = []
        if isinstance(u, str):
            assert '\uFE0F' not in u, f'Set: Variation Selector fail: {u}'

        self._ = OrderedDict.fromkeys(
            u._ if isinstance(u, Set)
                else u.split() if isinstance(u, str)
                    else u
        )

    def replace(self, u, v):
        return Set(str(self).replace(str(u), str(v)))

    def __iadd__(self, u):
        self._ = (self | Set(u))._
        return self

    def __or__(self, u):
        s = Set(self)
        for element in Set(u):
            s._[element] = None
        return s

    def __ror__(self, u):
        return Set(u) | self

    def __isub__(self, u):
        for element in Set(u):
            self._.pop(element, None)
        return self

    def __iter__(self):
        return iter(self._)

    def __len__(self):
        return len(self._)

    def __sub__(self, u):
        return Set([x for x in self if x not in Set(u)])

    def __repr__(self):
        return f'{" ".join(k for k in self._.keys())}'

    def __eq__(self, u):
        return list(self._.keys()) == list(Set(u)._)

    def __mod__(self, u):
        return set(self._) == set(Set(u)._)

    def __contains__(self, u):
        if isinstance(u, Set):
            return all(item in self._ for item in u)
        return u in self._

    def __and__(self, u):
        return Set([x for x in self if x in Set(u)])

