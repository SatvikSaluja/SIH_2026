import { useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { FeatureCollection, Feature } from 'geojson';
import type { Layer, PathOptions, LatLngBoundsExpression } from 'leaflet';
import { ParcelProperties } from '../utils/parcelGenerator';

interface MicroParcelMapProps {
  bounds: [[number, number], [number, number]];
  geoData: FeatureCollection | null;
}

function MapBoundsController({ bounds }: { bounds: [[number, number], [number, number]] }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.flyToBounds(bounds as LatLngBoundsExpression, { padding: [20, 20], duration: 1.2 });
    }
  }, [bounds, map]);
  return null;
}

export function MicroParcelMap({ bounds, geoData }: MicroParcelMapProps) {
  const styleFeature = (feature?: Feature): PathOptions => {
    const props = feature?.properties as ParcelProperties | undefined;
    if (!props) return { color: '#3388ff', weight: 1 };

    let fillColor = '#94a3b8'; // Not tracked (grey) -- geocadastra has no ownership concept, honestly distinct from "Private"
    if (props.ownership_status === 'Private') fillColor = '#0ea5e9'; // Blue
    if (props.ownership_status === 'Government') fillColor = '#f59e0b'; // Amber
    if (props.ownership_status === 'Disputed') fillColor = '#ef4444'; // Red

    return {
      fillColor,
      fillOpacity: 0.5,
      color: fillColor,
      weight: 1,
      opacity: 0.8,
    };
  };

  const onEachFeature = (feature: Feature, layer: Layer) => {
    const props = feature.properties as ParcelProperties;
    if (!props) return;

    const popupHtml = `
      <div style="font-family: system-ui, sans-serif; padding: 4px; min-width: 180px;">
        <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 14px;">${props.ulpin}</h4>
        <div style="font-size: 11px; color: #475569; line-height: 1.5;">
          <div><strong>Owner:</strong> ${props.owner_name ?? 'Not tracked'}</div>
          <div><strong>Area:</strong> ${props.area_sqm.toLocaleString()} sq.m</div>
          <div><strong>Status:</strong> <span style="color: ${props.ownership_status === 'Disputed' ? '#ef4444' : '#0ea5e9'}">${props.ownership_status ?? 'Not tracked'}</span></div>
          <div><strong>Confidence:</strong> ${props.confidence_score != null ? props.confidence_score + '%' : 'Not tracked'}</div>
        </div>
      </div>
    `;
    layer.bindPopup(popupHtml);

    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({ weight: 3, fillOpacity: 0.8, color: '#0f172a' });
        target.bringToFront();
      },
      mouseout: (e) => {
        const target = e.target;
        target.setStyle(styleFeature(feature));
      },
    });
  };

  return (
    <div style={{ height: '100%', width: '100%', borderRadius: '6px', overflow: 'hidden' }}>
      <MapContainer
        bounds={bounds as LatLngBoundsExpression}
        zoom={12}
        style={{ height: '100%', width: '100%', background: '#0f172a' }}
      >
        <MapBoundsController bounds={bounds} />
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key=cb1_3jbp_1_e0b02bc43cd98e6f50437e5f"
        />
        {geoData && (
          <GeoJSON
            key={geoData.features.length}
            data={geoData}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
    </div>
  );
}
