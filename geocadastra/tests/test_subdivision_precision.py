"""Boundary shape, not just total area, must survive subdivision."""
import hashlib
from dataclasses import asdict
import numpy as np
import pytest
import shapely
from shapely.geometry import box
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.synth.subdivision import recursive_partition, strip_partition
from geocadastra.jobs.orchestrator import _regenerate_ward


@pytest.mark.parametrize("seed", [0,1,7,21,42,123,300,301,302,660])
def test_new_subdivision_has_no_internal_crack_boundaries(seed):
    ward=generate_ward(seed=seed)
    for block in ward.blocks:
        parcels=[p.polygon for p in ward.parcels if p.block_id==block.id]
        union=shapely.union_all(parcels)
        assert union.boundary.hausdorff_distance(block.polygon.boundary)<1e-6
        assert abs(sum(p.area for p in parcels)-union.area)<1e-6
        assert union.symmetric_difference(block.polygon,grid_size=1e-9).area<1e-5


def test_old_job_regenerates_the_original_vector_and_raster_bytes():
    params=asdict(WardParams(generator_version=1))
    params.pop("generator_version")
    ward=_regenerate_ward("synthetic:seed=21",params)
    digest=hashlib.sha256()
    for parcel in ward.parcels:digest.update(parcel.polygon.wkb)
    for array in [ward.ortho,ward.dsm,ward.dtm]:digest.update(array.tobytes())
    assert digest.hexdigest()=="434bfc00cd548f4944affee6bef65a8d5c0aa23a4f183bc65a0a422201556ed1"
    assert ward.params.generator_version==1


def test_new_recursive_split_still_honors_its_work_budget():
    parts=recursive_partition(box(0,0,100,100),np.random.default_rng(1),.001,25,budget=12)
    assert 1<len(parts)<=13
    assert shapely.union_all(parts).equals(box(0,0,100,100))


def test_rotated_strip_cutters_preserve_the_world_space_domain():
    from shapely.affinity import rotate
    domain=rotate(box(0,0,100,20),27)
    parts=strip_partition(domain,8,np.random.default_rng(1))
    assert len(parts)==8
    assert shapely.union_all(parts).boundary.hausdorff_distance(domain.boundary)<1e-6
