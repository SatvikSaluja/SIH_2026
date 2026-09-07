"""Stage 5: disagreement detection and conflict records."""
import pytest
from shapely.geometry import Point

from geocadastra.core.conflicts import ConflictRecord, max_pairwise_disagreement
from geocadastra.core.crs import Geom
from geocadastra.core.fusion import SourceEstimate


def test_max_pairwise_disagreement_single_estimate_is_zero():
    assert max_pairwise_disagreement([SourceEstimate("legacy", 0.0, 0.0, 1.0)]) == 0.0


def test_max_pairwise_disagreement_is_the_largest_pairwise_distance():
    estimates = [
        SourceEstimate("legacy", 0.0, 0.0, 1.0),
        SourceEstimate("gt", 3.0, 4.0, 0.1),  # 5.0 from legacy
        SourceEstimate("model", 1.0, 0.0, 0.5),  # 1.0 from legacy, sqrt(4^2+4^2)=5.66 from gt
    ]
    assert max_pairwise_disagreement(estimates) == pytest.approx(5.0)  # legacy<->gt: sqrt(3^2+4^2)


def test_conflict_record_carries_parcel_ids_sources_magnitude_and_geometry():
    rec = ConflictRecord(
        node_id=7,
        face_ids=(1, 2),
        sources=("legacy", "gt"),
        disagreement_m=5.0,
        geometry=Geom(Point(0.0, 0.0), "EPSG:32643"),
    )
    assert rec.node_id == 7
    assert rec.face_ids == (1, 2)
    assert rec.sources == ("legacy", "gt")
    assert rec.disagreement_m == 5.0
    assert rec.geometry.crs == "EPSG:32643"
