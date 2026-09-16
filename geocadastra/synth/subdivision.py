"""Subdivision from one noded cut network, rather than snapped child polygons."""
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, split
from shapely.errors import GEOSException

PRECISION = 1e-9


def _partition(poly, cuts):
    if not cuts:
        return [poly]
    network = shapely.union_all([poly.boundary, *cuts], grid_size=PRECISION)
    domain = poly.buffer(PRECISION * 10)
    return [face for face in polygonize(network) if face.area > 0 and domain.covers(face.representative_point())]


def strip_partition(poly, n, rng):
    if n <= 1:
        return [poly]
    coords = list(poly.minimum_rotated_rectangle.exterior.coords)
    longest = int(np.argmax([Point(coords[i]).distance(Point(coords[i+1])) for i in range(4)]))
    p0, p1 = coords[longest], coords[longest+1]
    angle = np.degrees(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
    center = poly.centroid
    rotated = affinity.rotate(poly, -angle, origin=center)
    minx,miny,maxx,maxy = rotated.bounds
    raw = np.sort(rng.uniform(.08,.92,size=n-1))
    fractions = np.clip(np.diff(np.concatenate(([0.],raw,[1.]))),.05,None)
    fractions /= fractions.sum()
    xs = minx + np.cumsum(fractions)[:-1] * (maxx-minx)
    pad = max(maxy-miny,1.)
    # Rotate cutters, never rotate quantized parcels back into world space.
    cuts = [affinity.rotate(LineString([(x,miny-pad),(x,maxy+pad)]),angle,origin=center) for x in xs]
    return _partition(poly,cuts)


def recursive_partition(poly, rng, min_area, max_depth, budget=4096):
    cuts = []
    remaining = [budget]

    def visit(piece, depth):
        if depth >= max_depth or piece.area <= min_area*1.6 or remaining[0] <= 0:
            return
        minx,miny,maxx,maxy = piece.bounds
        diagonal = np.hypot(maxx-minx,maxy-miny)
        cx = rng.uniform(minx+(maxx-minx)*.3,minx+(maxx-minx)*.7)
        cy = rng.uniform(miny+(maxy-miny)*.3,miny+(maxy-miny)*.7)
        angle = rng.uniform(0,np.pi)
        dx,dy = np.cos(angle)*diagonal,np.sin(angle)*diagonal
        cut = LineString([(cx-dx,cy-dy),(cx+dx,cy+dy)])
        try:
            children = [g for g in split(piece,cut).geoms if g.geom_type=="Polygon" and g.area>0]
            if len(children)<2:
                return
            clipped_cut = cut.intersection(piece)
        except GEOSException:
            return
        cuts.append(clipped_cut)
        remaining[0] -= 1
        for child in children:
            visit(child,depth+1)

    visit(poly,0)
    return _partition(poly,cuts)
