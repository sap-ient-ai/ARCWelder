from types import FunctionType, LambdaType

from numpy.lib.stride_tricks import as_strided

from Set import Set
from Coord import Coord
from Coords import Coords
from Rect import Rect
from Iter import Iter
from Grid import Grid

class Supergrid:
    # TODO: tidy up gridlines and ε
    def __init__(self, G, sy=None, sx=None, ny=None, nx=None, gridlines=False):
        self.G = G
        self.ε = int(gridlines)

        none = lambda *args: all(item is None for item in args)

        if none(ny, nx, sy, sx):
            sy, sx = self._infer_cell_hw

        if none(sy, sx):
            ε = int(gridlines)
            sy, sx = (G.hw + ε) // Coord(ny, nx) - ε

        self.cell_hw = Coord(sy, sx)

    @property
    def _infer_cell_hw(self):
        # Here we assume gridlines are present.
        G = self.G
        H, W = G.hw
        for hw in Coords(G.rect + 1).sort():
            h, w = hw
            if  (H + 1) % (h + 1) == 0 and \
                (W + 1) % (w + 1) == 0:
                glyphs_y = G[   h::h+1].glyphs if h < H else Set()
                glyphs_x = G[:, w::w+1].glyphs if w < W else Set()
                if len(glyphs_y | glyphs_x) == 1:
                    self.ε = 1
                    return hw
        self.ε = 0
        raise ValueError('Failed to infer gridlines!')

    @property
    def np_subgrids(self):
        A, ε = self.G._, self.ε
        (Y, X), (y, x) = self.super_hw, self.cell_hw

        r, c = A.strides
        return as_strided(
            A,
            shape=(Y, X, y, x),
            strides=(
                r * (y + ε),
                c * (x + ε),
                r, c
            )
        )

    @property
    def super_hw(self):
        return (self.G.hw + self.ε) // (self.cell_hw + self.ε)

    @property
    def iter(self):
        return Iter(self)

    @property
    def superpixels(self):
        def pixel(p):
            H = Grid(self.np_subgrids[p.y][p.x])
            assert len(H.glyphs) == 1
            return str(H.glyphs)

        H = Grid(hw=self.super_hw, fill=self.G.dtype(0))
        for p in H.locs:
            H[p] = pixel(p)

        return H

    @superpixels.setter
    def superpixels(self, v):
        assert isinstance(v, Grid) and v.hw == self.super_hw
        for p in v.locs:
            self[p] = v[p]

    def __setitem__(self, k, v):
        K = Grid(hw=self.super_hw, fill=False) ;  K[k] = True
        for p in K.pix:
            self[p][:] = v

    def __getitem__(self, k):
        _k = Coord(k)
        return Grid(self.np_subgrids[_k.y, _k.x])

    def __iter__(self):
        return Iter(
            self[p].with_meta(p=p)
                for p in Coords(Rect(hw=self.super_hw))
        )

    @property
    def glyph(self):
        return self.G[self.cell_hw] if self.ε else None

    @property
    def stats(self):
        # if self.ε:
        N = len(self.iter.unique)
        uniform_state = 'each uniform' if all(self.iter.is_uniform) else 'non-uniform cells exist'

        return f'Gridlines={self.glyph} {self.super_hw:~x} of {self.cell_hw:~x}, {N} unique cells, {uniform_state}'

    def __eq__(self, u):
        U = Grid(u)
        G = Grid(hw=self.super_hw, fill=False)
        for p in G.locs:
            G[p] = self[p] @ U
        return G

    def __rmatmul__(self, f):
        assert isinstance(f, (FunctionType, LambdaType))
        f_dtype = type(f(self[0, 0]))
        G = Grid(hw=self.super_hw, fill=f_dtype(0))
        for p in G.locs:
            G[p] = f(self[p])
        return G
