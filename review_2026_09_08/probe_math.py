import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
from rasterio.transform import from_origin
from geocadastra.core.conformal import calibrate,certified_band
from geocadastra.core.fusion import legacy_estimate,model_estimate
from geocadastra.core.priority import face_uncertainty,simulate_survey,CostModel
from geocadastra.core.crs import Geom
from geocadastra.core.graph import build_graph
from geocadastra.core.transport import assign_parcels,parcels_to_graph
from geocadastra.core.evaluate import topology_validity_rate

def test_math_probes():
 rng=np.random.default_rng(42);pairs=rng.random((10000,2));hits=[]
 for a,b in pairs:hits.append(b<=certified_band(calibrate([('formal',a)],alpha=.1),'formal',1))
 print('n=1 nominal 90% empirical coverage:',np.mean(hits))
 poly=box(0,0,10,10); est=legacy_estimate((5,5),unary_union([poly]),'formal');boundary=legacy_estimate((5,5),poly.boundary,'formal')
 print('legacy polygon estimate:',est,'true boundary estimate:',boundary)
 graph=build_graph([Geom(poly,'EPSG:32643')],'EPSG:32643')
 print('unmeasured uncertainty:',face_uncertainty(graph,{}))
 print('unmeasured certified fraction:',simulate_survey([],{}, {},{0:tuple(graph.edges)},{0:(5,5)},(0,0),.1,CostModel(1,10))[0].certified_fraction)
 # One parcel, oblique block: polygonization drifts from the input area.
 block=Geom(Polygon([(0,0),(20,0),(0,20)]),'EPSG:32643')
 result=assign_parcels(block,[block.area],[(5,5)],np.zeros((40,40),np.float32),from_origin(0,20,.5,.5),n_segments=20)
 gg=parcels_to_graph(result.parcel_polygons,block)
 output=unary_union([g.geom for g in gg.faces_to_polygons().values()])
 print('one-parcel triangular block: input area',block.area,'output area',output.area,'faces',len(gg.faces),'symmetric difference',output.symmetric_difference(block.geom).area,'conflicts',[c.kind for c in result.conflicts])
 raw=20.0
 print('sigma trained vs interpreted:',np.exp(10*np.tanh(raw/10)/2),np.exp(raw/2))
 print('mean block medians vs true stratum median:',np.mean([np.median([0,0,0]),np.median([100])]),np.median([0,0,0,100]))
 # An invalid face is exactly what this evaluator should be able to count.
 invalid=build_graph([Geom(Polygon([(0,0),(2,2),(0,2),(2,0)]),'EPSG:32643'),Geom(box(.2,.2,1,1),'EPSG:32643')],'EPSG:32643')
 try:print('invalid topology metric:',topology_validity_rate(invalid))
 except Exception as e:print('invalid topology metric raised:',type(e).__name__)
