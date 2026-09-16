"""Prioritisation and the headline metric (Stage 7).

The headline number for this whole project is not IoU -- it is
surveyor-hours required to certify 90% of a ward, and the curve of
certified fraction against hours spent. This module builds the priority
order a field survey should follow to make that curve as steep as
possible, and simulates all three comparison curves (priority order,
random order, naive lowest-confidence-first order) so the claim "our
ordering reaches 90% in materially fewer hours" is a measured number, not
an assertion.

Pipeline: `build_parcel_adjacency()`/`face_uncertainty()`/`face_edge_ids()`/
`face_centroids()` (all per-graph -- parcels sharing an edge, per-edge
certified bands from Stage 6 aggregated to a per-parcel worst-edge number,
each parcel's own edge ids, and centroids; a multi-block ward merges
several graphs' worth of these into one ward-wide picture, since
prioritisation is inherently ward-scale even though the planar graph
itself is block-scoped) -> `priority_score()` (uncertainty weighted by how
much uncertain neighbourhood a parcel anchors) -> `cluster_parcels()`
(group the top-ranked ones spatially -- a surveyor visits a place, not a
database row) -> `rank_clusters()` (by total value over travel cost) ->
`simulate_survey()` (walk the resulting order, price it with `CostModel`,
propagate each visit's edge resolution to still-unvisited neighbours,
emit the certified-fraction-vs-hours curve). `build_priority_order()`
chains the scoring/clustering/ranking steps into the one sequence
`simulate_survey()` needs; `random_order()`/`lowest_confidence_order()`
build the two baseline sequences the real order must beat.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from geocadastra.core.graph import OUTER, PlanarGraph


@dataclass(frozen=True)
class CostModel:
    minutes_per_parcel: float  # fixed on-site time to physically verify one parcel
    travel_speed_m_per_min: float  # surveyor's effective travel speed between stops

    def travel_minutes(self, a: tuple, b: tuple) -> float:
        dx, dy = a[0] - b[0], a[1] - b[1]
        return (dx**2 + dy**2) ** 0.5 / self.travel_speed_m_per_min


@dataclass(frozen=True)
class SurveyPoint:
    hours: float
    certified_fraction: float


def build_parcel_adjacency(graph: PlanarGraph) -> dict[int, set[int]]:
    """Two parcels (faces) are adjacent iff they share an edge. Read
    directly off `graph.faces_of_edge()` rather than recomputed from
    geometry -- the planar graph (Stage 1) already *is* this adjacency
    information, just indexed by edge instead of by face pair.
    """
    adjacency: dict[int, set[int]] = {fid: set() for fid in graph.faces}
    for edge_id in graph.edges:
        f0, f1 = graph.faces_of_edge(edge_id)
        if f0 != OUTER and f1 != OUTER and f0 != f1:
            adjacency[f0].add(f1)
            adjacency[f1].add(f0)
    return adjacency


def _band(values: dict, key: int) -> float:
    value = values.get(key, math.inf)
    if math.isnan(value) or value < 0:
        raise ValueError("uncertainty must be nonnegative (infinity means unknown)")
    return value


def face_uncertainty(graph: PlanarGraph, edge_uncertainty: dict[int, float]) -> dict[int, float]:
    """Per-face uncertainty = the worst (max) of its own boundary edges'
    uncertainty. A parcel is not certified until every one of its edges is
    inside the department's tolerance, so averaging across edges would
    hide its worst edge behind its better ones -- max is the only
    aggregation consistent with "the parcel's own certified status".

    Missing bands are unbounded: a face requires every edge to be measured.
    """
    result = {}
    for fid, face in graph.faces.items():
        bands = [_band(edge_uncertainty, eid) for eid, _ in face.all_edges]
        result[fid] = max(bands, default=math.inf)
    return result


def priority_score(adjacency: dict[int, set[int]], uncertainty: dict[int, float]) -> dict[int, float]:
    """score(p) = uncertainty(p) * (1 + sum of its neighbours' own
    uncertainty) -- "uncertainty weighted by centrality" made concrete.

    Verifying parcel p resolves p's shared boundary with each neighbour
    too (Stage 1's own invariant: a shared edge is one object, not two
    independent copies) -- so a parcel bordering several still-uncertain
    neighbours is worth more to survey than an equally-uncertain parcel
    sitting in an otherwise-settled block, even though the two have
    identical uncertainty on their own. Summing neighbours' uncertainty
    (rather than e.g. plain degree centrality, which only counts how many
    neighbours exist, not how uncertain they are) is what ties the score
    to *which* neighbours a parcel anchors, per the doc's own phrasing.
    """
    return {
        fid: (0.0 if _band(uncertainty, fid) == 0 else
              _band(uncertainty, fid) * (1.0 + sum(_band(uncertainty, n) for n in neighbours)))
        for fid, neighbours in adjacency.items()
    }


def cluster_parcels(parcel_ids: list, centroids: dict, max_distance: float) -> list:
    """Single-linkage spatial clustering: group parcels transitively
    connected by a chain of pairwise distances <= `max_distance`. A
    surveyor is dispatched to a place, not to a database row -- two
    top-ranked parcels ten metres apart belong on the same stop.

    ponytail: O(n^2) pairwise distances -- fine for a "top-ranked" subset
    (hundreds of parcels), not a whole ward; a spatial index
    (scipy.spatial.cKDTree, already in the pinned stack via scipy) is the
    upgrade if that subset ever gets large enough for this to matter.
    """
    parent = {pid: pid for pid in parcel_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(parcel_ids):
        ax, ay = centroids[a]
        for b in parcel_ids[i + 1 :]:
            bx, by = centroids[b]
            if (ax - bx) ** 2 + (ay - by) ** 2 <= max_distance**2:
                union(a, b)

    groups: dict = defaultdict(list)
    for pid in parcel_ids:
        groups[find(pid)].append(pid)
    return list(groups.values())


def _group_centroid(group: list, centroids: dict) -> tuple:
    xs = [centroids[p][0] for p in group]
    ys = [centroids[p][1] for p in group]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def rank_clusters(clusters: list, scores: dict, centroids: dict, depot: tuple, cost_model: CostModel) -> list:
    """Rank whole clusters by total value over travel cost, not parcels
    individually -- a cluster with modest per-parcel scores but ten
    parcels on one street is worth more than a single high-score parcel
    that costs a whole trip on its own.

    The cost used here is a self-contained per-cluster estimate (travel
    from a fixed `depot`, e.g. the ward entry point) used only to RANK
    clusters -- `simulate_survey()` walks the resulting order and prices
    the real, sequential point-to-point route separately.

    ponytail: ranks by an independent depot-distance estimate rather than
    solving the true routing problem (a full tour's value/cost is
    order-dependent, a TSP variant); a real routing solver is the upgrade
    if travel cost turns out to dominate survey time in practice.
    """

    def value_over_cost(cluster: list) -> float:
        value = sum(scores.get(p, 0.0) for p in cluster)
        centroid = _group_centroid(cluster, centroids)
        cost = len(cluster) * cost_model.minutes_per_parcel + cost_model.travel_minutes(depot, centroid)
        return value / cost if cost > 0 else float("inf")

    return sorted(clusters, key=value_over_cost, reverse=True)


def face_edge_ids(graph: PlanarGraph) -> dict:
    """Every face's own boundary edge ids (Stage 1's `Face.boundary`,
    direction stripped) -- the shared-edge structure `simulate_survey()`
    needs to propagate a field visit's resolution to a parcel's still-
    unvisited neighbours: physically surveying one parcel establishes
    ground truth along its whole perimeter, which is the SAME graph edge
    (Stage 1's own invariant: a shared boundary is one object, not two
    independent per-parcel copies) a neighbouring parcel also depends on.
    """
    return {fid: tuple(eid for eid, _ in face.all_edges) for fid, face in graph.faces.items()}


def face_centroids(graph: PlanarGraph) -> dict:
    """Every face's centroid, keyed the same way `build_parcel_adjacency()`
    and `face_uncertainty()` key their own output -- the third piece
    `build_priority_order()` needs from a graph, split out separately so a
    multi-block caller can merge several graphs' worth of these (adjacency,
    uncertainty, centroids) into one ward-wide picture before prioritising,
    rather than this module assuming a single graph covers a whole ward.
    Blocks are this project's unit of topology (Stage 1's own invariant);
    prioritisation is inherently ward-scale (a surveyor's route isn't
    confined to one block) -- merging happens one level up, not here.
    """
    return {fid: (graph.face_polygon(fid).centroid.x, graph.face_polygon(fid).centroid.y) for fid in graph.faces}


def build_priority_order(
    adjacency: dict,
    uncertainty: dict,
    centroids: dict,
    depot: tuple,
    cost_model: CostModel,
    cluster_distance: float,
    tolerance: float,
) -> list:
    """The real recipe: score every parcel, restrict to the ones actually
    worth a field visit (`uncertainty > tolerance` -- a parcel already
    inside tolerance needs no survey time at all), cluster those
    spatially, and rank the resulting clusters by total value over travel
    cost.

    Every parcel needing verification gets the clustering/routing
    treatment, not just an arbitrary top slice with a lower-quality
    fallback for the rest -- an earlier version here capped clustering to
    a fixed `top_fraction` and fell back to plain lowest-confidence order
    for everything else, which defeats the point whenever most of the
    actual verification workload (the bulk of informal-settlement parcels
    here, by construction) falls in that unclustered "rest": the travel-
    time savings that make the priority order beat the naive baseline
    only apply to the part that got clustered. Simpler and correct is
    "cluster everything that needs a visit."

    Takes already-built `adjacency`/`uncertainty`/`centroids` (see
    `build_parcel_adjacency()`/`face_uncertainty()`/`face_centroids()`)
    rather than a single graph, so a caller can merge several blocks'
    worth of these into one ward-wide priority order.
    """
    scores = priority_score(adjacency, uncertainty)
    candidates = [pid for pid in adjacency if _band(uncertainty, pid) > tolerance]
    clusters = cluster_parcels(candidates, centroids, cluster_distance)
    # visit each cluster's own highest-scoring (most neighbour-anchoring)
    # parcels first: `cluster_parcels()`'s grouping order is arbitrary
    # (dict-iteration order of `candidates`), not scored -- without this,
    # a cluster could visit its lowest-value member first and its actual
    # hub last, delaying exactly the free neighbour-certifications
    # (simulate_survey()`) that centrality-weighting exists to front-load
    clusters = [sorted(c, key=lambda pid: scores[pid], reverse=True) for c in clusters]
    return rank_clusters(clusters, scores, centroids, depot, cost_model)


def random_order(parcel_ids: list, rng) -> list:
    """Baseline: survey in a random sequence, no clustering, no scoring --
    the naive floor the real priority order is compared against."""
    ids = list(parcel_ids)
    rng.shuffle(ids)
    return [[pid] for pid in ids]


def lowest_confidence_order(uncertainty: dict) -> list:
    """Baseline: survey the least-confident (most uncertain) parcel first,
    every time -- the single-signal ordering the Stage 7 doc names
    explicitly as what the real priority order must beat. No clustering,
    no centrality, no travel-cost awareness, purely uncertainty-sorted.
    """
    return [[pid] for pid, _ in sorted(uncertainty.items(), key=lambda kv: kv[1], reverse=True)]


def simulate_survey(
    order: list,
    adjacency: dict,
    edge_uncertainty: dict,
    face_edges: dict,
    centroids: dict,
    depot: tuple,
    tolerance: float,
    cost_model: CostModel,
) -> list:
    """Walk `order` (a sequence of parcel groups -- clusters, or singleton
    groups for an unclustered baseline) point to point, in the sequence
    each group already lists its own members in, and return the
    certified-fraction-vs-hours curve: this is the project's actual
    headline metric.

    Travel is priced between EVERY consecutive stop actually visited, not
    once per group to a group centroid -- an earlier version here charged
    one `travel_minutes()` call per group and treated every other member
    as reachable for free, which silently made a single large cluster
    (e.g. from single-linkage chaining across a dense block, see
    `cluster_parcels()`) collapse to near-zero internal travel cost
    regardless of how physically spread out its members actually were,
    an unrealistic surveyor route and, caught by actually testing this
    across several ward seeds, the entire source of "priority" beating
    the baselines rather than the intended centrality/clustering effects.

    Ground truth lives on EDGES, not parcels (`edge_uncertainty`,
    `face_edges` -- see `face_edge_ids()`): physically verifying parcel p
    resolves every one of p's boundary edges, which can also certify a
    still-unvisited NEIGHBOUR of p for free, if that shared edge was the
    thing keeping the neighbour's own worst-edge uncertainty above
    tolerance -- the real payoff behind weighting priority by centrality
    (see `priority_score()`), not merely a travel-time convenience. A
    parcel already at or under `tolerance` before any visits starts
    already certified; survey time -- and now a stop on the route -- is
    never spent on what's already trustworthy or already resolved by an
    earlier neighbour's visit.

    `edge_uncertainty` is copied internally, not mutated in place --
    a caller's own dict (e.g. one reused to simulate a second `order`)
    is left untouched.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    total = len(face_edges)
    if total == 0:
        return []
    edge_uncertainty = dict(edge_uncertainty)

    def current_uncertainty(fid: int) -> float:
        # Every boundary requires a measured band before certification.
        edges = face_edges.get(fid, ())
        return max((_band(edge_uncertainty, eid) for eid in edges), default=math.inf)

    certified = {fid for fid in face_edges if current_uncertainty(fid) <= tolerance}
    curve = [SurveyPoint(hours=0.0, certified_fraction=len(certified) / total)]

    minutes = 0.0
    here = depot
    for group in order:
        for pid in group:
            if pid in certified:
                continue  # a neighbour's earlier visit already resolved every edge of this parcel too
            minutes += cost_model.travel_minutes(here, centroids[pid])
            here = centroids[pid]
            minutes += cost_model.minutes_per_parcel
            for eid in face_edges.get(pid, ()):
                edge_uncertainty[eid] = 0.0
            certified.add(pid)
            for neighbour in adjacency.get(pid, ()):
                if neighbour not in certified and current_uncertainty(neighbour) <= tolerance:
                    certified.add(neighbour)  # resolved as a side effect, no separate visit needed
            curve.append(SurveyPoint(hours=minutes / 60.0, certified_fraction=len(certified) / total))
    return curve


def hours_to_reach(curve: list, target_fraction: float) -> float | None:
    """First `hours` value in `curve` at which `certified_fraction` first
    reaches `target_fraction`, or `None` if the curve never gets there --
    the single number the Stage 7 Done-when criterion is actually about.
    """
    for point in curve:
        if point.certified_fraction >= target_fraction:
            return point.hours
    return None
