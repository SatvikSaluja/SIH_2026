"""Coupled recorded-area refinement of an existing shared-edge graph.

The raster assignment supplies an evidence-aligned topology. This constrained
optimization changes node positions, not face identity or adjacency. Only a
validated, precision-snapped solution is returned as converged. An infeasible
or unsupported case retains its input graph and an explicit diagnostic.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from shapely import set_precision, union_all
from shapely.geometry import Point, LineString
from shapely.errors import GEOSException
from geocadastra.core.graph import Node
from geocadastra.core.planarize import GRID


@dataclass
class CapacityResult:
    graph: object
    converged: bool
    detail: dict


def refine_recorded_areas(graph, block, areas, tolerances, *, node_weights=None, max_iterations=120):
    if graph.crs != block.crs:
        from geocadastra.core.crs import CRSMismatchError
        raise CRSMismatchError("capacity refinement CRS mismatch")
    ids = sorted(areas)
    if not ids or any(not math.isfinite(areas[p]) or areas[p] <= 0 or
                      not math.isfinite(tolerances[p]) or tolerances[p] < 0 for p in ids):
        raise ValueError("recorded areas must be positive and tolerances nonnegative and finite")

    def declined(reason, **detail):
        return CapacityResult(graph,False,{"reason":reason,**detail})

    if set(graph.face_parcel_ids) != set(graph.faces) or set(graph.face_parcel_ids.values()) != set(ids):
        return declined("incomplete_parcel_identity")
    if abs(sum(areas.values())-block.area) > sum(tolerances.values()) + .01:
        return declined("infeasible_recorded_total",recorded_total=sum(areas.values()),block_area=block.area)
    work=deepcopy(graph)
    faces_by_parcel={pid:[fid for fid,p in graph.face_parcel_ids.items() if p==pid] for pid in ids}

    def parcel_areas():
        polygons=work.faces_to_polygons()
        return np.array([sum(polygons[fid].area for fid in faces_by_parcel[pid]) for pid in ids])

    target=np.array([areas[pid] for pid in ids])
    tolerance=np.array([tolerances[pid] for pid in ids])
    initial=parcel_areas()
    try:
        polygons=[g.geom for g in work.faces_to_polygons().values()]
        if not all(p.is_valid and p.area>0 for p in polygons):
            return declined("invalid_initial_face")
        union=union_all(polygons,grid_size=GRID)
        if (sum(p.area for p in polygons)-union.area>1e-4 or
                union.symmetric_difference(block.geom,grid_size=GRID).area>.01):
            return declined("initial_topology_not_tiled")
    except GEOSException:
        return declined("indeterminate_initial_topology")
    if np.all(np.abs(initial-target)<=tolerance):
        return CapacityResult(graph,True,{"reason":"already_within_tolerance","max_error_m2":float(np.max(np.abs(initial-target)))})

    domain=block.geom
    rings=[domain.exterior,*domain.interiors]
    vertices=[Point(c) for ring in rings for c in ring.coords[:-1]]
    segments=[LineString([a,b]) for ring in rings for a,b in zip(ring.coords[:-1],ring.coords[1:])]
    variables=[]
    x0=[]
    bounds=[]
    weights=[]
    minx,miny,maxx,maxy=domain.bounds
    for nid,node in graph.nodes.items():
        weight=float((node_weights or {}).get(nid,1.))
        if not math.isfinite(weight) or weight<=0:
            raise ValueError("node displacement weights must be positive and finite")
        point=Point(node.x,node.y)
        if point.distance(domain.boundary)<=2*GRID:
            if any(point.distance(vertex)<=2*GRID for vertex in vertices):
                continue  # preserve the authoritative block's corners
            segment=min(segments,key=lambda line:point.distance(line))
            variables.append((nid,segment))
            x0.append(segment.project(point))
            bounds.append((0.,segment.length))
            weights.append(weight)
        else:
            variables.append((nid,None))
            x0.extend([node.x,node.y])
            bounds.extend([(minx,maxx),(miny,maxy)])
            weights.extend([weight,weight])
    if not x0:
        return declined("no_movable_nodes")
    # Keep this dense local solver bounded. Larger blocks need a sparse solver,
    # not an unbounded quadratic allocation hidden in a worker.
    if len(x0)>256:
        return declined("dense_solver_size_limit",variables=len(x0))
    x0=np.array(x0)
    weights=np.array(weights)

    def apply(values):
        offset=0
        for nid,segment in variables:
            if segment is None:
                x,y=values[offset:offset+2]
                offset+=2
            else:
                point=segment.interpolate(values[offset])
                x,y=point.x,point.y
                offset+=1
            work.nodes[nid]=Node(nid,float(x),float(y))

    def residual(values):
        apply(values)
        # Total domain area is fixed by its ring; the last equality is
        # redundant. It is still checked independently after snapping.
        return ((parcel_areas()-target)/np.maximum(target,1.))[:-1]

    constraints=[] if len(ids)==1 else [{"type":"eq","fun":residual}]
    result=minimize(lambda x:float(np.sum(weights*(x-x0)**2)),x0,
                    jac=lambda x:2*weights*(x-x0),bounds=bounds,constraints=constraints,
                    method="SLSQP",options={"maxiter":max_iterations,"ftol":1e-12})
    if not np.all(np.isfinite(result.x)):
        return declined("optimizer_nonfinite")
    apply(result.x)
    # GEOS, rather than hand-rounded coordinates, defines the persisted grid.
    positions=set()
    for nid,node in work.nodes.items():
        point=set_precision(Point(node.x,node.y),GRID)
        xy=(point.x,point.y)
        if xy in positions:
            return declined("snapped_node_collision")
        positions.add(xy)
        work.nodes[nid]=Node(nid,*xy)
    try:
        polygons=[g.geom for g in work.faces_to_polygons().values()]
        if not all(p.is_valid and p.area>0 for p in polygons):
            return declined("invalid_refined_face")
        union=union_all(polygons,grid_size=GRID)
        if sum(p.area for p in polygons)-union.area>1e-4:
            return declined("refined_face_overlap")
        if union.symmetric_difference(domain,grid_size=GRID).area>.01:
            return declined("refined_block_coverage")
        errors=np.abs(parcel_areas()-target)
        if not np.all(errors<=tolerance):
            return declined("recorded_areas_not_met_after_snapping",max_error_m2=float(errors.max()),
                            optimizer_message=str(result.message))
    except GEOSException:
        return declined("indeterminate_refined_topology")
    from geocadastra.core.graph import _round_pt
    work._point_to_node={_round_pt((n.x,n.y)):nid for nid,n in work.nodes.items()}
    return CapacityResult(work,True,{"reason":"refined","max_error_m2":float(errors.max()),
                                    "initial_max_error_m2":float(np.abs(initial-target).max()),
                                    "iterations":int(result.nit)})
