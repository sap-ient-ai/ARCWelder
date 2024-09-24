from Grid import Grid

from Set import Set

from Iter import Iter

class Pair(Iter):
    def __init__(self, X, Y, i=None):
        self.X, self.Y = Grid(X, index=i), Grid(Y, index=i)
        self.index = i

        super().__init__([self.X, self.Y])

        self.refresh()

    def refresh(self):
        def get_op(bX, bY):
            return 'fixed' if bX.same(bY)  \
                else 'created' if bX.none  \
                    else 'removed' if bY.none   \
                        else 'extended' if (bY[bX]).all  \
                            else 'reduced' if (bX[bY]).all  \
                                else 'changed'

        if self.X.hw != self.Y.hw:
            self.glyphs = self.X.glyphs | self.Y.glyphs
            return

        self.glyph_op = {
            '🔳': get_op(self.X != 0, self.Y != 0)
        } | {
            u: get_op(self.X == u, self.Y == u)
                for u in self.X.glyphs | self.Y.glyphs
        }

        glyphs_for_op = lambda op: Set([
            glyph_ for glyph_, op_ in self.glyph_op.items()
                if op_ == op
        ])

        self.op_glyphs = { op: glyphs_for_op(op)
            for op in 'removed fixed created extended reduced changed'.split()
        }

        self.glyphs = Set(' '.join(map(str, self.op_glyphs.values()))) - '🔳'

        for k, v in self.op_glyphs.items():
            setattr(self, k, str(v) if len(v) == 1 else v)


    def overview(self):
        def stats_for_glyphs(G, glyphs):
            def stats_for(glyph):
                H = G[glyph]
                return f'{glyph}{H.npix:>3}px ' + ' '.join(
                    f'{s}{(H[s + '.#']):<2}'
                        for s in ['⛓', '🔗']
                )

            return ',  '.join(
                stats_for(glyph) if glyph in ('🔳' | G.glyphs)  \
                    else ' ' * 16
                        for glyph in glyphs
            )

        self.refresh()

        glyphs = '🔳' | self.glyphs
        index = f'[{self.index}]' if self.index is not None else ''

        for s, U in zip('XY', self):
            print(f'{s}{index}: {U.hw:~x5}  {stats_for_glyphs(U, glyphs)}')

        if self.X.hw == self.Y.hw:
            print(', '.join(
                f'{glyphs} {op}'
                    for op, glyphs in self.op_glyphs.items()
                        if glyphs
            ))

    @property
    def copy(self):
        return Pair(self.X.copy, self.Y.copy)