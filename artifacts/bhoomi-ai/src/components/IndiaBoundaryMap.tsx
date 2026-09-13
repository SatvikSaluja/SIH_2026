import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup, useMap } from 'react-leaflet';
import { RotateCcw, Maximize2, Layers } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import type { FeatureCollection, Feature } from 'geojson';
import type { Layer, PathOptions, LatLngBoundsExpression, Map as LeafletMap } from 'leaflet';

export interface StateRegionProperties {
  id: string;
  state_name: string;
  state_code: string;
  area_sqkm: number;
  parcel_count: number;
  topology_errors: number;
  encroachments: number;
  confidence_score: number;
  bounds: [[number, number], [number, number]];
  center: [number, number];
}

export interface InferenceMarkerData {
  patchId: string;
  regionName: string;
  parcelCount: number;
  confidence: number;
  coordinates: [number, number]; // [lat, lng]
}

interface IndiaBoundaryMapProps {
  selectedRegionId?: string;
  onSelectRegion?: (regionId: string) => void;
  onGeoDataLoaded?: (features: Feature[]) => void;
  inferenceMarker?: InferenceMarkerData | null;
}

function MapController({
  selectedRegionId,
  geoData,
  mapRef,
  inferenceMarker,
}: {
  selectedRegionId?: string;
  geoData: FeatureCollection | null;
  mapRef: React.MutableRefObject<LeafletMap | null>;
  inferenceMarker?: InferenceMarkerData | null;
}) {
  const map = useMap();

  useEffect(() => {
    mapRef.current = map;
  }, [map, mapRef]);

  useEffect(() => {
    if (inferenceMarker?.coordinates) {
      map.flyTo(inferenceMarker.coordinates, 8.5, { duration: 1.5 });
      return;
    }

    if (!geoData || !selectedRegionId) {
      map.flyTo([22.5937, 78.9629], 4.5, { duration: 1.2 });
      return;
    }

    const matchedFeature = geoData.features.find(
      (f) =>
        f.properties?.id === selectedRegionId ||
        f.properties?.state_name?.toLowerCase() === selectedRegionId.toLowerCase()
    );

    if (matchedFeature && matchedFeature.properties?.bounds) {
      const bounds = matchedFeature.properties.bounds as LatLngBoundsExpression;
      map.flyToBounds(bounds, { duration: 1.4, padding: [30, 30] });
    }
  }, [selectedRegionId, geoData, map, inferenceMarker]);

  return null;
}

export function IndiaBoundaryMap({
  selectedRegionId,
  onSelectRegion,
  onGeoDataLoaded,
  inferenceMarker,
}: IndiaBoundaryMapProps) {
  const [geoData, setGeoData] = useState<FeatureCollection | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);

  useEffect(() => {
    fetch('/data/india_boundaries.geojson')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load GeoJSON: ${res.statusText}`);
        }
        return res.json();
      })
      .then((data: FeatureCollection) => {
        setGeoData(data);
        setLoading(false);
        if (onGeoDataLoaded) {
          onGeoDataLoaded(data.features);
        }
      })
      .catch((err) => {
        console.error('Error fetching India GeoJSON:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const defaultStyle = (feature?: Feature): PathOptions => {
    const isSelected =
      feature &&
      selectedRegionId &&
      (feature.properties?.id === selectedRegionId ||
        feature.properties?.state_name?.toLowerCase() === selectedRegionId.toLowerCase());

    if (isSelected) {
      return {
        fillColor: '#059669',
        fillOpacity: 0.55,
        color: '#047857',
        weight: 3.8,
        opacity: 1,
      };
    }

    return {
      fillColor: '#0f172a',
      fillOpacity: 0.1,
      color: '#059669',
      weight: 1.8,
      opacity: 0.9,
    };
  };

  const hoverStyle: PathOptions = {
    fillColor: '#10b981',
    fillOpacity: 0.38,
    color: '#047857',
    weight: 3.0,
    opacity: 1,
  };

  const onEachFeature = (feature: Feature, layer: Layer) => {
    const props = feature.properties || {};
    const stateName =
      props.state_name ||
      props.STNAME_SH ||
      props.STNAME ||
      props.NAME_1 ||
      props.NAME ||
      'Indian Territory';

    const stateId = props.id || stateName.toLowerCase().replace(/\s+/g, '-');
    const area = props.area_sqkm ? props.area_sqkm.toLocaleString() : 'N/A';
    const parcels = props.parcel_count ? props.parcel_count.toLocaleString() : 'N/A';
    const code = props.state_code || 'IN';

    layer.on({
      mouseover: (e) => {
        const target = e.target;
        const isSelected =
          selectedRegionId &&
          (props.id === selectedRegionId ||
            props.state_name?.toLowerCase() === selectedRegionId.toLowerCase());

        if (!isSelected) {
          target.setStyle(hoverStyle);
        }
        if (target.bringToFront) {
          target.bringToFront();
        }
      },
      mouseout: (e) => {
        const target = e.target;
        target.setStyle(defaultStyle(feature));
      },
      click: () => {
        console.log(`Region Clicked: ${stateName} (${stateId})`, props);
        if (onSelectRegion) {
          onSelectRegion(stateId);
        }
      },
    });

    layer.bindTooltip(`<strong>${stateName}</strong><br/><small>LGD: ${code} · ${area} km²</small>`, {
      sticky: true,
      className: 'custom-leaflet-tooltip',
    });

    const popupHtml = `
      <div style="font-family: system-ui, -apple-system, sans-serif; padding: 6px; min-width: 210px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
          <h4 style="margin: 0; color: #0f172a; font-size: 15px; font-weight: 700;">${stateName}</h4>
          <span style="background: #e2e8f0; color: #334155; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;">LGD ${code}</span>
        </div>
        <div style="font-size: 11px; color: #475569; line-height: 1.6; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 6px 0; margin-bottom: 8px;">
          <div><strong>Area:</strong> ${area} km²</div>
          <div><strong>Est. Parcels:</strong> ${parcels}</div>
          <div><strong>Topology Errors:</strong> ${props.topology_errors ?? 0}</div>
          <div><strong>Model Confidence:</strong> ${props.confidence_score ?? 98.5}%</div>
        </div>
        <button 
          id="btn-select-${stateId}"
          style="width: 100%; background: #059669; color: white; border: none; border-radius: 4px; padding: 7px 10px; font-size: 11px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 4px; box-shadow: 0 2px 5px rgba(5,150,105,0.3);"
          onclick="window.dispatchEvent(new CustomEvent('bhoomi:select-region', { detail: '${stateId}' })); this.closest('.leaflet-popup') && document.querySelectorAll('.leaflet-popup-close-button').forEach(function(b){b.click()});"
        >
          ⚡ Select Jurisdiction
        </button>
      </div>
    `;

    layer.bindPopup(popupHtml);
  };

  useEffect(() => {
    const handleCustomSelect = (e: Event) => {
      const customEvent = e as CustomEvent<string>;
      if (customEvent.detail && onSelectRegion) {
        onSelectRegion(customEvent.detail);
        if (mapRef.current) {
          mapRef.current.closePopup();
        }
      }
    };
    window.addEventListener('bhoomi:select-region', handleCustomSelect);
    return () => {
      window.removeEventListener('bhoomi:select-region', handleCustomSelect);
    };
  }, [onSelectRegion]);

  const presets = [
    { label: 'All India', id: '' },
    { label: 'Delhi', id: 'delhi' },
    { label: 'Maharashtra', id: 'maharashtra' },
    { label: 'Tamil Nadu', id: 'tamil-nadu' },
    { label: 'Rajasthan', id: 'rajasthan' },
  ];

  return (
    <div className="map-canvas live-map-canvas" data-testid="map-parcel-awareness">
      {/* Floating HUD Overlay */}
      <div className="map-hud-overlay">
        <div className="preset-chips-wrap">
          {presets.map((p) => (
            <button
              key={p.label}
              className={`preset-chip ${
                (selectedRegionId === p.id || (!selectedRegionId && p.id === '')) ? 'preset-chip-active' : ''
              }`}
              onClick={() => onSelectRegion && onSelectRegion(p.id)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="map-action-btns">
          <button
            className="map-action-btn"
            title="Reset to All India View"
            onClick={() => onSelectRegion && onSelectRegion('')}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {loading && (
        <div className="map-loading-overlay">
          <span>Loading Indian Vector Boundaries...</span>
        </div>
      )}
      {error && (
        <div className="map-error-overlay">
          <span>Failed to load map data: {error}</span>
        </div>
      )}

      <MapContainer
        center={[22.5937, 78.9629]}
        zoom={4.5}
        scrollWheelZoom={false}
        style={{ width: '100%', height: '100%', borderRadius: '6px', background: '#f8fafc' }}
      >
        <MapController selectedRegionId={selectedRegionId} geoData={geoData} mapRef={mapRef} inferenceMarker={inferenceMarker} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {geoData && (
          <GeoJSON
            key={selectedRegionId || 'all-india'}
            data={geoData}
            style={defaultStyle}
            onEachFeature={onEachFeature}
          />
        )}

        {/* Predicted Patch Location Red Circle Marker */}
        {inferenceMarker && inferenceMarker.coordinates && (
          <>
            <CircleMarker
              center={inferenceMarker.coordinates}
              radius={24}
              pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.25, weight: 1.5 }}
            />
            <CircleMarker
              center={inferenceMarker.coordinates}
              radius={12}
              pathOptions={{ color: '#ffffff', fillColor: '#ef4444', fillOpacity: 0.95, weight: 3 }}
            >
              <Popup>
                <div style={{ fontFamily: 'system-ui, sans-serif', padding: '6px', minWidth: '220px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyBetween: 'space-between', marginBottom: '6px', borderBottom: '1px solid #e2e8f0', paddingBottom: '4px' }}>
                    <strong style={{ color: '#ef4444', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      📍 Predicted Patch Location
                    </strong>
                  </div>
                  <div style={{ fontSize: '12px', color: '#1e293b', lineHeight: '1.6' }}>
                    <div>Jurisdiction: <strong>{inferenceMarker.regionName}</strong></div>
                    <div>Patch ID: <strong>{inferenceMarker.patchId}</strong></div>
                    <div>Extracted Parcels: <strong>{inferenceMarker.parcelCount.toLocaleString()}</strong></div>
                    <div>Model Confidence: <strong>{inferenceMarker.confidence}%</strong></div>
                    <div style={{ marginTop: '8px', padding: '6px 8px', background: '#fef2f2', borderRadius: '4px', fontFamily: 'monospace', fontWeight: 700, color: '#dc2626', border: '1px solid #fca5a5' }}>
                      Map Coordinates:<br />
                      {inferenceMarker.coordinates[0].toFixed(5)}° N, {inferenceMarker.coordinates[1].toFixed(5)}° E
                    </div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          </>
        )}
      </MapContainer>

      <div className="map-scale">
        <span>0</span>
        <i />
        <span>500 km</span>
      </div>
      <div className="map-label label-north">N</div>
      <div className="map-label label-zone">
        {selectedRegionId ? `Selected: ${selectedRegionId.toUpperCase()}` : 'India / All Jurisdictions'}
      </div>
    </div>
  );
}
