"""Roads -> blocks (Stage 1).

Roads partition a ward into blocks: planarize the road linework against the
ward's own boundary (passed as authoritative/fixed, so it can attract
nearby road endpoints without ever moving itself -- see planarize.py), and
take the resulting faces as blocks. The ward boundary needs no other special
handling: every block edge that only borders one real block automatically
gets `OUTER` as its other face-side from graph.py's own sentinel, which *is*
"the ward boundary as the outer face."
"""
from __future__ import annotations

import shapely

from geocadastra.core.crs import Geom
from geocadastra.core.graph import PlanarGraph, build_graph
from geocadastra.core.planarize import GRID, planarize


def build_blocks(road_linework: list[Geom], ward_boundary: Geom, crs: str, grid: float = GRID) -> PlanarGraph:
    """`road_linework` (LineString Geoms) + `ward_boundary` (a Polygon Geom)
    -> a PlanarGraph whose faces are blocks. Road segments outside the ward
    boundary are ignored -- they polygonize into faces that don't overlap
    the ward and are filtered out, along with any degenerate sliver face."""
    boundary_line = Geom(ward_boundary.geom.boundary, crs)
    faces = planarize(road_linework, grid=grid, fixed=[boundary_line])

    ward_buffered = ward_boundary.geom.buffer(grid)
    shapely.prepare(ward_buffered)
    min_area = grid * grid
    inside = [f for f in faces if f.geom.area > min_area and ward_buffered.contains(f.geom.representative_point())]
    return build_graph(inside, crs, collinear_tol=grid)
