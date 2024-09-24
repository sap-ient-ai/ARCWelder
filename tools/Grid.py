# TODO:
#   - .npix -> Σ
#   - Ensure Grid('🔴') gives a 2d grid
#   - aperture = box['0.🟡.✂.rect'] in Y CANNOT
#   - .update ???
#   - Allow `Coord in Grid`? currently need Grid.rect

# region 🦋 🦋 imports 🦋 🦋

import numpy as np

# TODO: put all scipy stuff somewhere (separate region)
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes, convolve

from numpy.lib.stride_tricks import as_strided

from Set import Set
from Coord import Coord
from Coords import Coords
from Rect import Rect

from Iter import Iter

# from functools import reduce
# import operator

# from collections.abc import Iterable
from numbers import Number

from Iter import Iter
from types import FunctionType, LambdaType

from collections import Counter

# from collections.abc import Iterable

# endregion


# region 🦋 🦋 Vars (GLYPHS, E, I) 🦋 🦋

# 🔳 is reserved for 'binarized, ⭕ for padding'
GLYPHS = '⚫ ⚪ 🔵 🟢 🟡 🟣 🟠 🟤 🔴 🌑    🔳 ⭕ 🌹'.split()
_I = {u: i for i, u in enumerate(GLYPHS)} | {'▪': False, '▫': True}

def I(e):
    return _I.get(e, e)

def E(i):
    if isinstance(i, np.generic):
        i = i.item()

    if i in _I.keys():
        return i
    assert isinstance(i, (bool, int))
    return '▪▫'[i] if isinstance(i, bool)  \
        else GLYPHS[i] if isinstance(i, int)  \
            else i

# endregion


# region🦋 🦋 functions:dumps _is_coord_as_tuple _is_integer) 🦋 🦋

def dumps(A, emojis=True, spaceVal=None):
    assert isinstance(A, np.ndarray)

    glyphs = ['▪', '▫'] if A.dtype == bool else GLYPHS

    spacing = 1 if emojis else max(A.flatten(), lambda u: len(str(u)))
    spaces = ' ' * spacing
    dumps_val = lambda v:  \
        ' ' if v == spaceVal  \
            else glyphs[v] if emojis  \
                else f'{v:{spacing + 1}}'
    dumps_row = lambda row: spaces.join(map(dumps_val, row))

    rows = [[A]] if A.ndim == 0 else [A] if A.ndim == 1 else A
    return '\n' + '\n'.join(map(dumps_row, rows)) + '\n'

def _is_coord_as_tuple(k):
    return isinstance(k, tuple)  \
        and len(k) == 2  \
            and all(isinstance(u, int) for u in k)

def _is_integer(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

# endregion


# 🦋 🦋 🦋 S4 🦋 🦋 🦋

class S4:
    def __init__(self, G):
        self.G = Grid(G)

    def __getitem__(self, key):
        if key.startswith('~'):
            key = S4.invert(key[1:])
        return self.symmetries[key]

    def __iter__(self):
        yield from self.symmetries.items()

    @staticmethod
    def invert(glyph):
        return {'⤴':'⤵', '⤵':'⤴'}.get(glyph, glyph)

    @property
    def symmetries(self):
        A = self.G._
        D = {
            '⏺': A,
            '⤴': np.rot90(A, 1),
            '🔄': np.rot90(A, 2),
            '⤵': np.rot90(A, 3),

            '↔': np.fliplr(A),
            '↕': np.flipud(A),
            '↘': A.T,
            '↗': np.fliplr(np.fliplr(A).T)
        }
        return { T: Grid(A).with_meta(T=T)
            for T, A in D.items()
        }

    def get_transforms_to(self, H, _exclude=''):
        return Set(k for k, v in self if v.same(H) and k not in _exclude)

    @property
    def self_symmetries(self):
        return self.get_transforms_to(self.G, _exclude='⏺')

    def get_transforms_from(self, H):
        return S4(H).get_transforms_to(self.G)

    @staticmethod
    def unique(grids):
        remaining = grids
        got = []
        while remaining:
            G, *others = remaining
            got.append(G)
            remaining = [ H for H in others
                if not S4(G).get_transforms_to(H)
            ]
        return got


# 🦋 🦋 🦋 Grid 🦋 🦋 🦋

def np_array_from_str(s):
    dtype = bool if '▪' in s or '▫' in s else int
    lines = s.replace(';', '\n').strip().split('\n')
    values_for_line = lambda line: [I(u) for u in line.replace(' ', '')]
    k = [values_for_line(u) for u in lines]
    return np.array(k, dtype=dtype)

def hxw_c(s):
    hxw, c = s.split('.')
    h, w = hxw.split('x')
    i = I(c)
    return np.full([int(h), int(w)], i, type(i))

class Grid:
    # region 🦋 __init__ copy int bool  🦋

    def __init__(
        self,
        k=None,
        hw=None, fill=None,
        **kwargs,
    ):
        if k is None:
            k = f'{Coord(hw):~x}.{E(fill)}'

        if isinstance(k, str):
            self._ = hxw_c(k) if '.' in k else np_array_from_str(k)

        elif isinstance(k, np.ndarray):
            self._ = k

        elif isinstance(k, Grid):
            self._ = k._

        else:
            self._ = np.array(k)

        self.meta = kwargs

    def with_meta(self, *args, **kwargs):
        if args:
            assert len(args) == 1
            G = args[0]
            assert isinstance(G, Grid)
            D = kwargs | G.meta
        else:
            D = kwargs
        self.meta |= D
        return self

    @property
    def dtype(self):
        t = self._.dtype.type
        dtype = {np.int64: int, np.bool_: bool}.get(t, t)
        assert dtype in (bool, int)
        return dtype

    @property
    def copy(self):
        return Grid(self._.copy(), **self.meta)

    @property
    def int(self):
        return Grid(self._.astype(int))

    @property
    def bool(self):
        return Grid(self._ != 0)

    # endregion


    # region 🦋 convenience properties: glyphs npix hw h w rect pix locs 🦋

    @property
    def glyphs(self):
        return Set(str(self))

    @property
    def is_uniform(self):
        return len(self.glyphs) == 1

    @property
    def npix(self):
        return int((self._ != 0).sum())

    @property
    def Σ(self):
        return self.npix

    @property
    def counts(self):
        s = str(self).replace(' ', '').replace('\n', '')
        return Counter(s).most_common()

    def most_common(self, exclude=''):
        exclude = exclude.replace('-', '').replace('~', '')
        return Iter(glyph for glyph, count in self.counts if glyph not in exclude)

    @property
    def hw(self):
        return Coord(self._.shape)

    @property
    def h(self):
        return self.hw.h

    @property
    def w(self):
        return self.hw.w

    @property
    def rect(self):
        # TODO: Probably Rect(self.hw) is enough
        return Rect(tl=(0, 0), hw=self.hw)

    @property
    def pix(self):
        return Coords(np.array(np.where(self._)).T)

    @property
    def locs(self):
        return Coords(Rect(self.rect))

    # endregion


    # region 🦋 __repr__ glyph_hxw_npix 🦋

    def __repr__(self):
        return dumps(self._)

    @property
    def glyph_hxw_npix(self):
        return f'{self.glyph} {self.hw:~x5} {self.npix:>3}px'

    # endregion


    # Convenience to avoid having to G.meta.frame etc. all the time. Just G.frame directly.
    def __getattr__(self, k):
        # Fix issue with IPython _foo attributes
        # https://stackoverflow.com/questions/78899267/
        # Rect.__getattr__ does the same

        if k.startswith('_'):
            return

        return self.meta[k] if k in self.meta.keys()  \
            else object.__getattribute__(self, k)


    # region 🦋 [ ] __getitem__ / __setitem__ 🦋

    def __getitem__(self, k):
        canvas = self.meta.get('canvas', None)
        frame = self.meta.get('frame', None)

        # 🔹 Rect
        if isinstance(k, Rect):
            if canvas:
                return canvas[k + frame.tl].with_meta(src=self, frame=k)
            else:
                return Grid(self._[k.slice], src=self)

        # 🔹 Coord
        if _is_coord_as_tuple(k):
            k = Coord(k)

        if isinstance(k, Coord):
            if canvas:
                return canvas[k + frame.tl]
            else:
                v = self.dtype(self._[k.y, k.x])
                return bool(v) if self.dtype == bool else E(v)

        # 🔹 Coords
        if isinstance(k, Coords):
            if canvas:
                return canvas[k + frame.tl]
            else:
                return Grid(self._[k.y.L, k.x.L])

        # 🔹 Grid (masking)
        if isinstance(k, Grid):
            assert (k.dtype, k._.shape) == (bool, self._.shape)
            return self[k._]

        # 🔹 str
        if isinstance(k, str):
            # MacOS Emoji-Palette seems to arbitrarily introduce these characters
            assert '\uFE0F' not in k, f'Grid: Variation Selector fail: {k}'

            # We will split off the righmost component R, recurse to calculate the left part, then apply R; we might be applying it to a single element or an Iter

            if ',' in k:
                return Iter(self[u.strip()] for u in k.split(','))

            bulk, _, last = k.rpartition('.')

            A = self[bulk] if bulk else self

            # A might be a Grid, an Iter of Grids, a Coord, a list of Coords

            if last == '#':
                return len(A)

            if _is_integer(last):  # index
                return A[int(last)]

            # Now for everything below, if A is a list, we have to do the op for every item and return a list

            def process(A):
                nonlocal last

                if last == '•':
                    return A.pix

                if last == '🖼':
                    return A.frame

                if last == '✂':
                    return A.crop

                if last == '🖨':
                    return A.copy


                negate = last.startswith('~')

                rots = ' ⏺ ⤴ 🔄 ⤵ '
                refs = ' ↔ ↕ ↘ ↗ '
                s4 = rots + refs

                last = last  \
                    .replace('卐', rots)  \
                    .replace('🪞', refs)  \
                    .replace('💠', s4)

                if (is_s4 := set(last) <= set('~' + s4)):
                    if negate:
                        assert len(last) == 2
                    S = Set(
                        S4.invert(last[1]) if negate else last
                    )
                    return A.s4[str(S)] if len(S) == 1  \
                        else Iter(A.s4[u] for u in S)

                if last[0] in ['⛓', '🔗']:
                    conn = last[0]
                    assert isinstance(A, Grid)
                    if A.dtype == bool:
                        L, n = label(
                            A._,
                            structure=generate_binary_structure(
                                rank=2,
                                connectivity={'⛓': 1, '🔗': 2}[conn]
                            )
                        )
                        return Iter(
                            Grid(L == k + 1).with_meta(A).crop
                                for k in range(n)
                        )
                    else:
                        # Need to flatten, as each glyph may have multiple regions
                        return Iter(
                            A[glyph][conn]
                                for glyph in A.glyphs - '⚫'
                        ).flatten


                _last = last
                if isinstance(A, Grid):
                    last = last.replace('…', str(A.glyphs - '⚫'))

                if '🔳' in last:
                    assert last == '🔳', 'Not implmented, do 🔳 separately!'
                    return (self != 0).with_meta(glyph='🔳')

                if '->' in last:
                    return A.send(last)

                if (ends_glyph := last[-1] in _I.keys()):
                    if last.startswith('~'):
                        assert len(last) == 2
                        return (A != last[1]).with_meta(glyph=last)
                    else:
                        return (A == last).with_meta(glyph=last) if len(last) == 1  \
                            else (lambda g: (A == g).with_meta(glyph=g)) @ Set(last)

                # TODO: How do we want to handle G['🔴🟣.🔗']?
                # Currently we're returning the 🔴regions and the 🟣regions
                # If we instead want to identify whether each pixel is in 🔴🟣 or not, and take regions over _this_, the below code gives a solution path.
                # But we should probably decide upon a syntax, e.g. G['{🔴,🟣}.🔗'] vs G['{🔴|🟣}.🔗']
                # And then we risk getting fancy, with expressions within our ''

                # if (is_glyphs := set(last) <= set(GLYPHS + ['~', ' '])):
                #     glyphs = self.glyphs - '⚫' - last[1:]  if negate  \
                #         else Set(last)
                #     O = operator
                #     comp, op = (O.ne, O.and_) if negate else (O.eq, O.or_)
                #     return Grid(
                #         reduce(op, (comp(self, u) for u in glyphs)),
                #     ).with_meta(A, glyph=_last)

                # fallback
                return getattr(A, last)

            return (process @ A).flatten  if isinstance(A, Iter)  \
                else process(A)

        # fallback; if we didn't detect any special case, use numpy's indexing

        return Grid(self._[k])

    def __setitem__(self, k, v):
        canvas = self.meta.get('canvas', None)
        frame = self.meta.get('frame', None)

        # TODO: Revise __setitem__ comment
        # 1. slice object

        # 2. Rect or Rect-like (G[1,2,3,4] which gives k=tuple(1,2,3,4))
        #       We use Rect.slice.
        #       If self.canvas exists, we translate coords to canvas-space and operate on the canvas. This allows G[-1, -1, 1, 1] = ...

        # 3a. Coord or Coord-like (e.g. G[1,2], G[(1, 2)]
        #       We promote to Coords (with ONE item) and proceed with 3

        # 3b. Coords or Coord-like (for now we allow Iter,
        #       Same self.canvas trick as 1.
        # TODO: Allow np.array of int, shape 2x{N})

        # 4. Mask (Grid with dtype=bool, same shape as self) or Mask-like (raw np.array)

        # 5. str
        #       Currently we require str is a glyph

        # 🔸 Ensure v is np.array or Number
        if isinstance(v, str):
            # TODO: examine 1D and 0D grids -- do we need/want a special case for v?
            # Or can we in all cases do v -> Grid(v)._.flatten()
            v = I(v) if len(v) == 1 else Grid(v)

        if isinstance(v, Grid):
            v = v._

        if not isinstance(v, (Number, np.ndarray)):
            v = np.array(v)


        # 🔸 Handle coordinates (Coord-like, Coord, Coords-like, Coords)

        if _is_coord_as_tuple(k):  k = Coord(k)
        if isinstance(k, Coord):   k = Coords([k])
        # if isinstance(k, Iter):    k = Coords(k)
        if isinstance(k, Coords):
            if canvas:
                canvas[k + frame.tl] = v
            else:
                self._[k.y.L, k.x.L] = v.flatten()  \
                    if isinstance(v, np.ndarray)  \
                        else v
            return

        # 🔸 Handle rectangles

        # 🔹 Rect-like -> Rect; allow G[1, 2, 3, 4]
        if isinstance(k, tuple) and len(k) == 4  \
            and all(isinstance(u, Number) for u in k):
                k = Rect(k)

        if isinstance(k, Rect):
            if canvas:
                canvas[k + frame.tl] = v
            else:
                k = k.slice
                self._[k] = v
            return


        # 🔸 Handle Grid

        # 🔹 Grid (fall-thru to Grid-like)
        if isinstance(k, Grid):
            k = k._

        # 🔸 Grid-like (masking ONLY)
        is_mask = isinstance(k, np.ndarray) and k.shape == self._.shape and k.dtype == bool
        if is_mask:
            self._[k] = v.flatten() if isinstance(v, np.ndarray) else v
            return

        # 🔸🔸🔸 Fallback to numpy

        # send Iter to List
        def listify(U):
            return [listify(u) for u in U] if isinstance(U, Iter)  \
                else U
        k = listify(k)

        # TODO: Now that we have upgraded Iter, maybe we can simplify getitem/setitem

        # def deepfix(U):
        #     if isinstance(U, Iterable):
        #         return [deepfix(u) for u in U]
        #     return U

        self._[k] = v

    # endregion


    # region 🦋 numpy wrappers: fill eye triu tril 🦋

    @property
    def rows(self):
        return Iter(
            Grid(r[np.newaxis, :])
                for r in self._
        )

    @property
    def cols(self):
        return Iter(
            Grid(c[:, np.newaxis])
                for c in self._.T
        )

    def fill(self, c, in_place=True):
        A = self if in_place else self.copy
        A[:] = I(c)
        return A

    @property
    def eye(self):
        assert self.h == self.w
        return Grid(np.eye(self.h).astype(self.dtype))

    @property
    def triu(self):
        assert self.h == self.w
        return Grid(np.triu(self._).astype(self.dtype))

    @property
    def tril(self):
        assert self.h == self.w
        return Grid(np.tril(self._).astype(self.dtype))

    # SCIPY stuff (TODO: new region for this?)

    @property
    def holes(self):
        assert self.dtype == bool
        return Grid(binary_fill_holes(self._)) - self

    def neighbours(self, conn='❇'):  # ❇ ❎ ✳
        kernel = {
            '❇': '''
                0 1 0
                1 0 1
                0 1 0
            ''',
            '❎': '''
                0 1 0
                1 0 1
                0 1 0
            ''',
            '✳': '''
                0 1 0
                1 0 1
                0 1 0
            ''',
        }[conn]
        kernel = Grid(kernel)._
        neighbor_count = convolve(self.int._, kernel, mode='constant', cval=0)
        return self * neighbor_count

    # endregion


    # region 🦋 boolean checks: same all none any 🦋

    def same(self, u):
        return np.array_equal(self._, Grid(u)._)

    @property
    def all(self):
        return bool(self._.all())

    @property
    def none(self):
        return bool((self._ == 0).all())

    @property
    def any(self):
        return bool(self._.any())

    # endregion


    # region 🦋 operator overloads: == != * | & ~ > |= in 🦋

    def __eq__(self, u):
        return Grid(self._ == Grid(u)._)

    def __ne__(self, u):
        return Grid(self._ != Grid(u)._)

    def __mul__(self, u):
        return Grid(self._ * Grid(u)._)

    def __or__(self, u):
        U = Grid(u)
        assert self.dtype == U.dtype, 'not mix types for |'
        return Grid(self._ | U._) if self.dtype == bool  \
            else self + (self['⚫']) * U

            # we want this to be like Python's `a or b`, so if for each pixel: out.pix = self.pix or u.pix
            # i.e. Only the ⚫ pixels of self get coloured to U

            # return Grid(self._ + (self._ == 0) * U._)

            # A = self._.copy()
            # A[A==0] = U._[A==0]
            # return Grid(A)

            # return self + (self == '⚫') * U

            # return self + (self['⚫']) * U

    def __ior__(self, u):
        U = Grid(u)
        assert self.dtype == U.dtype, 'not mix types for |'
        # self._[self._ != 0] = U._[self._ != 0]
        self._[:] = (self | U)._
        return self

    def __add__(self, u):
        return Grid(self._ + Grid(u)._)

    def __iadd__(self, u):
        self._ += Grid(u)._
        return self

    def __sub__(self, u):
        assert self.dtype == Grid(u).dtype == bool
        # self=0 -> 0
        # self=1 -> 0 if u=1 else 1
        return Grid(self._ & (self._ ^ u._))

    def __and__(self, u):
        # TODO: hmm, we could just .bool everything -- if the user wants .int masking they can use *
        return Grid(
            self._ & Grid(u)._  if self.dtype == Grid(u).dtype == bool
                else self.int * Grid(u).int
        )

    def __invert__(self):
        assert self.dtype == bool
        return Grid(~self._)


    def __contains__(self, u):
        assert E(u) in _I.keys()
        # TODO: gridA in gridB could be useful
        return E(u) in self.glyphs

    def __gt__(self, u):
        return self.s4.self_symmetries if isinstance(u, str) and u == '⏺'  \
            else self.s4.get_transforms_to(u) or '❌'

    # TODO: Think about whether we want NotImplemented or super.@
    def __matmul__(self, u):
        return self.same(u) if isinstance(u, Grid)  \
            else NotImplemented

    def __rmatmul__(self, u):
        return u(self) if isinstance(u, (FunctionType, LambdaType))  \
            else super().__rmatmul__(u)

    # endregion


    # region 🦋 s4 touchings roll pad shift crop split gen_subgrids locate 🦋

    @property
    def s4(self):
        return S4(self)

    @property
    def touchings(self):
        glyphs = [ glyph
            for glyph, f, F in zip('👆👇👈👉', self.frame.tblr, self.src.rect.tblr)
                if f == F
        ]
        return ' '.join(glyphs) or '🔲'

    def send(self, *args, in_place=False):
        "send('⚪ 🔵 -> ⚫'), send('⚪ 🔵', '⚫')"

        P, Q = '->'.join(args).replace(' ', '').split('->')
        if len(P) > len(Q):
            Q *= len(P)

        G = self.copy
        for p, q in zip(P, Q):
            G[self == p] = q

        if in_place:
            self[:] = G
            return self

        else:
            return G

    def roll(self, Δ, in_place=False):
        A = np.roll(self._, shift=Coord(Δ)._, axis=(0, 1))
        if in_place:
            self[:] = A
            return self
        else:
            return Grid(A, **self.meta)

    # Return a fresh Grid G with a G.canvas as a backing-Grid
    def pad(self, *args, padding=None, **kwargs):
        '''
        pad(1, '⭕')
        pad('⭕', 1)
        pad(tlr=(2, 2, 2))
        pad('▫', t=1, l=1)
        '''

        default_padding = {bool: '▪', int: '⚫'}[self.dtype]
        if (args_padding := str(Set(args) & Set(_I.keys()))):
            args = tuple(a for a in args if a != args_padding)
        pad = args_padding or padding or default_padding

        canvas_rect = self.rect.extend(*args, **kwargs)
        frame = self.rect - canvas_rect.tl

        canvas = Grid(hw=canvas_rect.hw, fill=pad)

        canvas[frame] = self

        return canvas[frame].with_meta(self, canvas=canvas, frame=frame)

    def shift(self, *delta, filler='⚫', in_place=False):
        Δ = Coord(*delta)
        A = self.pad(tl=Δ.abs, br=Δ.abs, padding=filler)
        A.canvas.roll(Δ, in_place=True)
        if in_place:
            self[:] = A
            return None
        else:
            return A.with_meta(self, canvas=None)

    # NOTE:
    # .frame is "frame-within-backing-grid"
    # - If G has .canvas, G.canvas[G.frame] recovers G
    # - If G has .src, G.src[G.frame] recovers G
    @property
    def crop(self):
        frame = self.pix.bounds
        return self[frame].with_meta(self, src=self, frame=frame)

    # This could be removed -- we could use Supergrid for this, but .split is nice and convenient.
    def split(self, ny=None, nx=None):
        if isinstance(ny, Coord):
            ny, nx = ny

        if ny is not None and nx is not None:
            return Iter( yy.split(nx=nx)
                for yy in self.split(ny=ny)
            )

        if ny is not None:
            return Iter( Grid(U)
                for U in self._.reshape(
                    (ny, self.h // ny,  self.w)
                )
            )

        if nx is not None:
            return Iter( Grid(U)
                for U in self._.reshape(
                    (self.h,  nx, self.w // nx)
                ).transpose(1, 0, 2)
            )

    def gen_subgrids(self, subgrid_shape, iterate=False):
        subgrid_shape = Coord(subgrid_shape)

        A = self._
        subgrid_dims = self.hw - subgrid_shape + 1
        if subgrid_dims.min < 1:
            raise ValueError('Subgrid dimensions are too large for the given grid size.')

        grids = as_strided(
            A,
            shape=[*subgrid_dims, *subgrid_shape],
            strides=A.strides + A.strides
        )
        return ( (index, grids[index])
                for index in np.ndindex(*subgrid_dims)
        ) if iterate  \
            else grids

    def locate(self, kernel):
        kernel = Grid(kernel)
        if kernel.w > self.w or kernel.h > self.h:
            raise ValueError(f'locate: kernel ({kernel.hw}) exceeds grid size ({self.hw})')
        grids = self.gen_subgrids(kernel.hw)
        counts = (grids == kernel._).sum(axis=(-1, -2))
        return Grid(counts == kernel._.size)

    # endregion

    def upscale(self, k):
        h, w = Coord(k)
        return Grid(
            self._.repeat(h, axis=0).repeat(w, axis=1)
        )
