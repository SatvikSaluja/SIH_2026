import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { api, type ParcelFeature } from "../api";

// The working CRS (EPSG:32643, this project's own metric UTM zone) has no
// business being reprojected through a slippy map's Web Mercator lat/lon --
// the /parcels endpoint already takes a plain metric bbox, so this map
// treats parcel coordinates as a flat plane (L.CRS.Simple) instead of
// pretending they are lon/lat. That is honest about what they are: metres.
interface Props {
  wardJobId: number;
  toleranceByParcel: Map<number, boolean>;
  onMapClick: (blockId: number, x: number, y: number) => void;
  refreshToken: number;
}

export function ParcelMap({ wardJobId, toleranceByParcel, onMapClick, refreshToken }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.GeoJSON | null>(null);
  // Leaflet's LatLngBounds.isValid() is true the instant any view has ever
  // been set -- which is immediately, since the map is constructed with one.
  // It cannot answer "has this map been fit to real data yet", so that has
  // to be tracked explicitly instead of inferred from the bounds object.
  const hasFitRef = useRef(false);
  const [status, setStatus] = useState<string>("loading");

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, { crs: L.CRS.Simple, minZoom: -6 }).setView([0, 0], 0);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  const load = async () => {
    const map = mapRef.current;
    if (!map) return;
    // L.CRS.Simple's [lat, lng] is [y, x] in the source plane -- bounds come
    // back in that order too, so this un-flips it back to (minx, miny, maxx, maxy).
    const b = map.getBounds();
    const bbox: [number, number, number, number] = [
      b.getWest() - 1000,
      b.getSouth() - 1000,
      b.getEast() + 1000,
      b.getNorth() + 1000,
    ];
    try {
      setStatus("loading");
      const { parcels } = await api.parcels(wardJobId, bbox);
      renderParcels(parcels);
      setStatus(`${parcels.length} parcels`);
    } catch (e) {
      setStatus(`error: ${(e as Error).message}`);
    }
  };

  const renderParcels = (parcels: ParcelFeature[]) => {
    const map = mapRef.current;
    if (!map) return;
    layerRef.current?.remove();

    // Coordinates are already metric (x, y); Leaflet with CRS.Simple wants
    // [y, x] pairs, so geometries are flipped on the way in, not projected.
    const flipped = {
      type: "FeatureCollection" as const,
      features: parcels.map((p) => ({
        type: "Feature" as const,
        properties: { face_id: p.face_id, block_id: p.block_id, parcel_id: p.parcel_id },
        geometry: flipGeometry(p.geometry),
      })),
    };

    const layer = L.geoJSON(flipped, {
      style: (feature) => {
        const parcelId = feature?.properties.parcel_id as number | null;
        const within = parcelId != null ? toleranceByParcel.get(parcelId) : undefined;
        const color = within === undefined ? "#94a3b8" : within ? "#16a34a" : "#dc2626";
        return { color, weight: 1, fillColor: color, fillOpacity: 0.25 };
      },
      onEachFeature: (feature, lyr) => {
        const { face_id, block_id, parcel_id } = feature.properties;
        lyr.bindTooltip(`face ${face_id} · block ${block_id} · parcel ${parcel_id ?? "unassigned"}`);
        // /parcels returns each face's resolved ring, never node ids -- a
        // click can only prefill a candidate (block_id, x, y); it must not
        // manufacture a node id that isn't real. The edit form still
        // requires an actual node id, sourced from /conflicts or field notes.
        lyr.on("click", (ev: L.LeafletMouseEvent) => {
          const latlng = ev.latlng;
          onMapClick(block_id, latlng.lng, latlng.lat);
        });
      },
    }).addTo(map);
    layerRef.current = layer;

    if (parcels.length && !hasFitRef.current) {
      hasFitRef.current = true;
      map.fitBounds(layer.getBounds());
    }
  };

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    // First load: there is no ward-extent endpoint, so centre on a generous
    // guess and let the user pan/zoom -- fitBounds() in renderParcels() then
    // snaps to the real extent the first time parcels actually come back.
    hasFitRef.current = false;
    map.setView([0, 0], -2);
    load();
    map.on("moveend", load);
    return () => {
      map.off("moveend", load);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wardJobId, refreshToken]);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.setStyle((feature) => {
        const parcelId = feature?.properties.parcel_id as number | null;
        const within = parcelId != null ? toleranceByParcel.get(parcelId) : undefined;
        const color = within === undefined ? "#94a3b8" : within ? "#16a34a" : "#dc2626";
        return { color, weight: 1, fillColor: color, fillOpacity: 0.25 };
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toleranceByParcel]);

  return (
    <div className="map-panel">
      <div ref={containerRef} className="map-container" />
      <div className="map-status">{status}</div>
      <div className="map-legend">
        <span><i style={{ background: "#16a34a" }} /> within tolerance</span>
        <span><i style={{ background: "#dc2626" }} /> outside tolerance</span>
        <span><i style={{ background: "#94a3b8" }} /> no recorded-area report</span>
      </div>
    </div>
  );
}

function flipGeometry(geom: GeoJSON.Geometry): GeoJSON.Geometry {
  const flip = (ring: number[][]): number[][] => ring.map(([x, y]) => [y, x]);
  if (geom.type === "Polygon") {
    return { type: "Polygon", coordinates: geom.coordinates.map(flip) };
  }
  if (geom.type === "MultiPolygon") {
    return { type: "MultiPolygon", coordinates: geom.coordinates.map((poly) => poly.map(flip)) };
  }
  return geom;
}
