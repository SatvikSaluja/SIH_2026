"""Reconstruct new-format graphs solely from committed changeset events.

No face cache, face-boundary rows, or current node/edge rows are consulted.
Old histories without an initial topology event are explicitly unsupported;
a migration must not fabricate evidence that was never recorded.
"""
from collections import defaultdict
from sqlalchemy import select
from geocadastra.core.graph import Edge, Face, Node, PlanarGraph, _round_pt
from geocadastra.store.schema import Changeset


def replay_block_graph(session, block_id: int, as_of_changeset: int | None = None) -> PlanarGraph:
    statement = select(Changeset).where(Changeset.affected_entities["block_id"].astext == str(block_id)).order_by(Changeset.id)
    if as_of_changeset is not None:
        statement = statement.where(Changeset.id <= as_of_changeset)
    graph = None
    for changeset in session.scalars(statement):
        event = changeset.affected_entities
        seed = event.get("graph_seed")
        if seed is not None:
            if graph is not None:
                raise ValueError("multiple initial topology events for the same block")
            graph = PlanarGraph(seed["crs"])
            graph.nodes = {nid:Node(nid,x,y) for nid,x,y in seed["nodes"]}
            graph.edges = {eid:Edge(eid,n0,n1,tuple(tuple(c) for c in coords)) for eid,n0,n1,coords in seed["edges"]}
            for fid,rings in seed["faces"]:
                walks = [tuple((eid,forward) for eid,forward in ring) for ring in rings]
                graph.faces[fid] = Face(fid,walks[0],tuple(walks[1:]))
        if graph is None:
            raise ValueError("history has no initial topology event; legacy replay requires an audited baseline")
        for nid,xy in event.get("node_positions",{}).items():
            nid = int(nid)
            if nid not in graph.nodes:
                raise ValueError(f"history moves unknown node {nid}")
            # The event already contains the exact committed snapped point.
            graph.nodes[nid] = Node(nid,*xy)
        graph.face_parcel_ids.update({int(fid):pid for fid,pid in event.get("face_parcel_ids",{}).items()})
    if graph is None:
        raise ValueError("no replayable history for block")
    edge_faces = defaultdict(list)
    for fid,face in graph.faces.items():
        for eid,_ in face.all_edges:
            edge_faces[eid].append(fid)
    graph._edge_faces = dict(edge_faces)
    graph._point_to_node = {_round_pt((n.x,n.y)):nid for nid,n in graph.nodes.items()}
    graph._next_node = max(graph.nodes,default=-1)+1
    graph._next_edge = max(graph.edges,default=-1)+1
    return graph
