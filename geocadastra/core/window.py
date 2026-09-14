"""The pixel window covering a map-space extent -- one implementation.

Reading a block's pixels off disk and cropping them out of an in-memory
ward are the same computation, and having it written twice let the two
disagree by a pixel: rasterio's `round_offsets().round_lengths()` floors the
offset and then ceils the *length*, which drops the partially-covered pixel
at the far edge, while flooring the near edge and ceiling the far edge keeps
it. A window that undercovers its block feeds the model a strip of missing
evidence along two sides, so the covering version is the correct one.
"""
from __future__ import annotations

import math


def pixel_window(transform, height: int, width: int, bounds):
    """`(row0, row1, col0, col1)` half-open, clamped, and never empty.

    Covers `bounds` (minx, miny, maxx, maxy) completely: any pixel the extent
    touches at all is included. Clamped to the raster because a block can sit
    flush against -- or past -- the ward edge, and never empty because a
    degenerate extent would otherwise round to zero width.
    """
    minx, miny, maxx, maxy = bounds
    inverse = ~transform
    corners = [inverse * (x, y) for x in (minx, maxx) for y in (miny, maxy)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    col0 = min(max(math.floor(min(cols)), 0), width - 1)
    row0 = min(max(math.floor(min(rows)), 0), height - 1)
    col1 = min(max(math.ceil(max(cols)), col0 + 1), width)
    row1 = min(max(math.ceil(max(rows)), row0 + 1), height)
    return row0, row1, col0, col1
