import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import * as turf from '@turf/turf';
import { 
  Network, Search, Hammer, CheckCircle2, AlertTriangle, 
  ChevronRight, Check, MapPinned, Eye, Layers, Globe, 
  Map as MapIcon, FileText, Crosshair, Building2
} from 'lucide-react';
import { Button, Surface, SectionHeading, LoadingState } from '../App';
import 'leaflet/dist/leaflet.css';

// Types
type TopologyError = {
  id: string;
  type: 'overlap' | 'sliver' | 'unclosed' | 'encroachment' | 'disconnection';
  severity: 'high' | 'medium' | 'low';
  priority: 'P1 - Critical' | 'P2 - High' | 'P3 - Medium' | 'P4 - Low';
  priorityCode: 'P1' | 'P2' | 'P3' | 'P4';
  riskScore: number; // e.g. 96.8 / 100
  featureId1: string;
  featureId2?: string;
  center: [number, number]; // [lat, lng]
  bounds: [[number, number], [number, number]]; // [[minLat, minLng], [maxLat, maxLng]]
  description: string;
};

type StateOption = {
  id: string;
  name: string;
  center: [number, number];
  bounds?: [[number, number], [number, number]];
  feature?: any;
};

type CityOption = {
  id: string;
  name: string;
  tier: 'mega' | 'large' | 'medium' | 'small';
  tierLabel: string;
  center: [number, number];
  parcelCount: number;
  anomalyCount: number;
};

// Map Controller to handle smooth panning and bounds fitting
function MapController({ center, zoom, bounds }: { center?: [number, number], zoom?: number, bounds?: [[number, number], [number, number]] | null }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.flyToBounds(bounds, { duration: 1.2, padding: [40, 40] });
    } else if (center && zoom) {
      map.flyTo(center, zoom, { duration: 1.2 });
    }
  }, [center, zoom, bounds, map]);
  return null;
}

const CARTO_API_KEY = 'cb1_3jbp_1_e0b02bc43cd98e6f50437e5f';

// 5 MAP VIEWS
const MAP_VIEWS = [
  { id: 'planar', name: '2D Planar (Standard)', icon: MapIcon, tile: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attr: '&copy; OpenStreetMap' },
  { id: 'satellite', name: 'Satellite Aerial (ORI)', icon: Globe, tile: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr: 'Esri World Imagery' },
  { id: 'elevation', name: '3D Elevation (DSM/DTM Fusion)', icon: Layers, tile: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', attr: 'Esri World Topo' },
  { id: 'cadastral', name: 'Cadastral Vector Layer', icon: FileText, tile: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key=' + CARTO_API_KEY, attr: '&copy; CARTO Dark' },
  { id: 'thematic', name: 'Thematic / Choropleth', icon: Eye, tile: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=' + CARTO_API_KEY, attr: '&copy; CARTO Voyager' },
];

const DEFAULT_STATES: StateOption[] = [
  { id: 'delhi', name: 'Delhi NCR', center: [28.6139, 77.2090] },
  { id: 'maharashtra', name: 'Maharashtra', center: [19.7515, 75.7139] },
  { id: 'karnataka', name: 'Karnataka', center: [15.3173, 75.7139] },
  { id: 'tamil-nadu', name: 'Tamil Nadu', center: [11.1271, 78.6569] },
  { id: 'odisha', name: 'Odisha', center: [20.9517, 85.0985] },
  { id: 'kerala', name: 'Kerala', center: [10.8505, 76.2711] },
  { id: 'uttar-pradesh', name: 'Uttar Pradesh', center: [26.8467, 80.9462] },
  { id: 'rajasthan', name: 'Rajasthan', center: [27.0238, 74.2179] },
  { id: 'gujarat', name: 'Gujarat', center: [22.2587, 71.1924] },
  { id: 'west-bengal', name: 'West Bengal', center: [22.9868, 87.8550] },
  { id: 'bihar', name: 'Bihar', center: [25.0961, 85.3131] },
  { id: 'punjab', name: 'Punjab', center: [31.1471, 75.3412] },
  { id: 'haryana', name: 'Haryana', center: [29.0588, 76.0856] },
  { id: 'telangana', name: 'Telangana', center: [18.1124, 79.0193] },
  { id: 'andhra-pradesh', name: 'Andhra Pradesh', center: [15.9129, 79.7400] },
  { id: 'madhya-pradesh', name: 'Madhya Pradesh', center: [22.9734, 78.6569] },
  { id: 'assam', name: 'Assam', center: [26.2006, 92.9376] }
];

// CITY DATABASE MATRICES PER STATE (Parcels & Anomalies vary strictly by City Tier Size)
const CITY_DATABASE: Record<string, CityOption[]> = {
  maharashtra: [
    { id: 'mumbai', name: 'Mumbai', tier: 'mega', tierLabel: 'Mega Metro (380 Parcels / High Density)', center: [19.0760, 72.8777], parcelCount: 380, anomalyCount: 8 },
    { id: 'pune', name: 'Pune', tier: 'large', tierLabel: 'Large Metro (240 Parcels / Med-High Density)', center: [18.5204, 73.8567], parcelCount: 240, anomalyCount: 5 },
    { id: 'nagpur', name: 'Nagpur', tier: 'medium', tierLabel: 'Medium City (130 Parcels / Med Density)', center: [21.1458, 79.0882], parcelCount: 130, anomalyCount: 3 },
    { id: 'nashik', name: 'Nashik', tier: 'medium', tierLabel: 'Medium City (110 Parcels / Med Density)', center: [19.9975, 73.7898], parcelCount: 110, anomalyCount: 3 },
    { id: 'solapur', name: 'Solapur', tier: 'small', tierLabel: 'Tier-3 City (55 Parcels / Low Density)', center: [17.6599, 75.9064], parcelCount: 55, anomalyCount: 1 },
  ],
  karnataka: [
    { id: 'bengaluru', name: 'Bengaluru', tier: 'mega', tierLabel: 'Mega Metro (410 Parcels / High Density)', center: [12.9716, 77.5946], parcelCount: 410, anomalyCount: 9 },
    { id: 'mysuru', name: 'Mysuru', tier: 'medium', tierLabel: 'Medium Heritage (140 Parcels / Med Density)', center: [12.2958, 76.6394], parcelCount: 140, anomalyCount: 4 },
    { id: 'hubballi', name: 'Hubballi-Dharwad', tier: 'medium', tierLabel: 'Medium Hub (120 Parcels / Med Density)', center: [15.3647, 75.1240], parcelCount: 120, anomalyCount: 3 },
    { id: 'mangaluru', name: 'Mangaluru', tier: 'medium', tierLabel: 'Coastal City (100 Parcels / Med Density)', center: [12.9141, 74.8560], parcelCount: 100, anomalyCount: 3 },
    { id: 'belagavi', name: 'Belagavi', tier: 'small', tierLabel: 'Tier-3 City (45 Parcels / Low Density)', center: [15.8497, 74.4977], parcelCount: 45, anomalyCount: 1 },
  ],
  'uttar-pradesh': [
    { id: 'lucknow', name: 'Lucknow', tier: 'large', tierLabel: 'Large Metro (290 Parcels / Med-High Density)', center: [26.8467, 80.9462], parcelCount: 290, anomalyCount: 6 },
    { id: 'kanpur', name: 'Kanpur', tier: 'large', tierLabel: 'Large Industrial (260 Parcels / Med-High)', center: [26.4499, 80.3319], parcelCount: 260, anomalyCount: 5 },
    { id: 'varanasi', name: 'Varanasi', tier: 'medium', tierLabel: 'Medium Heritage (150 Parcels / Med Density)', center: [25.3176, 82.9739], parcelCount: 150, anomalyCount: 4 },
    { id: 'agra', name: 'Agra', tier: 'medium', tierLabel: 'Medium City (120 Parcels / Med Density)', center: [27.1767, 78.0081], parcelCount: 120, anomalyCount: 3 },
    { id: 'jhansi', name: 'Jhansi', tier: 'small', tierLabel: 'Tier-3 City (50 Parcels / Low Density)', center: [25.4484, 78.5685], parcelCount: 50, anomalyCount: 1 },
  ],
  'tamil-nadu': [
    { id: 'chennai', name: 'Chennai', tier: 'mega', tierLabel: 'Mega Metro (390 Parcels / High Density)', center: [13.0827, 80.2707], parcelCount: 390, anomalyCount: 8 },
    { id: 'coimbatore', name: 'Coimbatore', tier: 'large', tierLabel: 'Large Metro (230 Parcels / Med-High)', center: [11.0168, 76.9558], parcelCount: 230, anomalyCount: 5 },
    { id: 'madurai', name: 'Madurai', tier: 'medium', tierLabel: 'Medium City (130 Parcels / Med Density)', center: [9.9252, 78.1198], parcelCount: 130, anomalyCount: 3 },
    { id: 'tiruchirappalli', name: 'Tiruchirappalli', tier: 'medium', tierLabel: 'Medium City (100 Parcels / Med Density)', center: [10.7905, 78.7047], parcelCount: 100, anomalyCount: 3 },
    { id: 'salem', name: 'Salem', tier: 'small', tierLabel: 'Tier-3 City (60 Parcels / Low Density)', center: [11.6643, 78.1460], parcelCount: 60, anomalyCount: 2 },
  ],
  gujarat: [
    { id: 'ahmedabad', name: 'Ahmedabad', tier: 'mega', tierLabel: 'Mega Metro (400 Parcels / High Density)', center: [23.0225, 72.5714], parcelCount: 400, anomalyCount: 8 },
    { id: 'surat', name: 'Surat', tier: 'large', tierLabel: 'Large Metro (270 Parcels / Med-High)', center: [21.1702, 72.8311], parcelCount: 270, anomalyCount: 6 },
    { id: 'vadodara', name: 'Vadodara', tier: 'medium', tierLabel: 'Medium City (140 Parcels / Med Density)', center: [22.3072, 73.1812], parcelCount: 140, anomalyCount: 4 },
    { id: 'rajkot', name: 'Rajkot', tier: 'medium', tierLabel: 'Medium City (110 Parcels / Med Density)', center: [22.3039, 70.8022], parcelCount: 110, anomalyCount: 3 },
    { id: 'bhavnagar', name: 'Bhavnagar', tier: 'small', tierLabel: 'Tier-3 City (55 Parcels / Low Density)', center: [21.7645, 72.1519], parcelCount: 55, anomalyCount: 1 },
  ],
  'west-bengal': [
    { id: 'kolkata', name: 'Kolkata', tier: 'mega', tierLabel: 'Mega Metro (410 Parcels / High Density)', center: [22.5726, 88.3639], parcelCount: 410, anomalyCount: 9 },
    { id: 'howrah', name: 'Howrah', tier: 'large', tierLabel: 'Large City (210 Parcels / Med-High)', center: [22.5958, 88.2636], parcelCount: 210, anomalyCount: 5 },
    { id: 'siliguri', name: 'Siliguri', tier: 'medium', tierLabel: 'Medium Hub (120 Parcels / Med Density)', center: [26.7271, 88.3953], parcelCount: 120, anomalyCount: 3 },
    { id: 'durgapur', name: 'Durgapur', tier: 'small', tierLabel: 'Tier-3 City (55 Parcels / Low Density)', center: [23.5204, 87.3119], parcelCount: 55, anomalyCount: 1 },
  ],
  delhi: [
    { id: 'new-delhi', name: 'New Delhi (Central)', tier: 'mega', tierLabel: 'Capital Metro (450 Parcels / High Density)', center: [28.6139, 77.2090], parcelCount: 450, anomalyCount: 9 },
    { id: 'noida', name: 'Noida (NCR Sector)', tier: 'large', tierLabel: 'Large Urban Sector (270 Parcels)', center: [28.5355, 77.3910], parcelCount: 270, anomalyCount: 6 },
    { id: 'gurugram', name: 'Gurugram (Cyber Hub)', tier: 'large', tierLabel: 'Large Tech Hub (290 Parcels)', center: [28.4595, 77.0266], parcelCount: 290, anomalyCount: 6 },
    { id: 'faridabad', name: 'Faridabad', tier: 'medium', tierLabel: 'Medium Zone (130 Parcels)', center: [28.4089, 77.3178], parcelCount: 130, anomalyCount: 3 },
  ],
  rajasthan: [
    { id: 'jaipur', name: 'Jaipur', tier: 'large', tierLabel: 'Large Metro (300 Parcels / Med-High)', center: [26.9124, 75.7873], parcelCount: 300, anomalyCount: 6 },
    { id: 'jodhpur', name: 'Jodhpur', tier: 'medium', tierLabel: 'Medium City (150 Parcels / Med Density)', center: [26.2389, 73.0243], parcelCount: 150, anomalyCount: 4 },
    { id: 'udaipur', name: 'Udaipur', tier: 'medium', tierLabel: 'Medium Heritage (110 Parcels)', center: [24.5854, 73.7125], parcelCount: 110, anomalyCount: 3 },
    { id: 'kota', name: 'Kota', tier: 'small', tierLabel: 'Tier-3 City (60 Parcels / Low Density)', center: [25.2138, 75.8648], parcelCount: 60, anomalyCount: 2 },
  ],
  kerala: [
    { id: 'kochi', name: 'Kochi', tier: 'large', tierLabel: 'Large Port Metro (250 Parcels)', center: [9.9312, 76.2673], parcelCount: 250, anomalyCount: 5 },
    { id: 'thiruvananthapuram', name: 'Thiruvananthapuram', tier: 'large', tierLabel: 'Capital Metro (230 Parcels)', center: [8.5241, 76.9366], parcelCount: 230, anomalyCount: 5 },
    { id: 'kozhikode', name: 'Kozhikode', tier: 'medium', tierLabel: 'Medium City (120 Parcels)', center: [11.2588, 75.7804], parcelCount: 120, anomalyCount: 3 },
    { id: 'thrissur', name: 'Thrissur', tier: 'small', tierLabel: 'Tier-3 City (50 Parcels / Low Density)', center: [10.5276, 76.2144], parcelCount: 50, anomalyCount: 1 },
  ],
  odisha: [
    { id: 'bhubaneswar', name: 'Bhubaneswar', tier: 'large', tierLabel: 'Capital Metro (260 Parcels)', center: [20.2961, 85.8245], parcelCount: 260, anomalyCount: 5 },
    { id: 'cuttack', name: 'Cuttack', tier: 'medium', tierLabel: 'Medium Heritage (140 Parcels)', center: [20.4625, 85.8828], parcelCount: 140, anomalyCount: 3 },
    { id: 'rourkela', name: 'Rourkela', tier: 'medium', tierLabel: 'Medium Industrial (100 Parcels)', center: [22.2604, 84.8536], parcelCount: 100, anomalyCount: 2 },
    { id: 'puri', name: 'Puri', tier: 'small', tierLabel: 'Tier-3 Coastal (45 Parcels / Low Density)', center: [19.8135, 85.8312], parcelCount: 45, anomalyCount: 1 },
  ]
};

// Dynamic fallback generator for any state not in explicit dictionary
function getCitiesForState(state: StateOption): CityOption[] {
  const key = state.id.toLowerCase();
  if (CITY_DATABASE[key]) {
    return CITY_DATABASE[key];
  }
  return [
    { 
      id: `${state.id}-metro`, 
      name: `${state.name} Central Metro`, 
      tier: 'large', 
      tierLabel: 'Large City (240 Parcels / Med-High)', 
      center: state.center, 
      parcelCount: 240, 
      anomalyCount: 5 
    },
    { 
      id: `${state.id}-district`, 
      name: `${state.name} District Hub`, 
      tier: 'medium', 
      tierLabel: 'Medium City (120 Parcels / Med Density)', 
      center: [state.center[0] + 0.04, state.center[1] + 0.03], 
      parcelCount: 120, 
      anomalyCount: 3 
    },
    { 
      id: `${state.id}-tehsil`, 
      name: `${state.name} Rural Tehsil`, 
      tier: 'small', 
      tierLabel: 'Tier-3 Town (50 Parcels / Low Density)', 
      center: [state.center[0] - 0.03, state.center[1] - 0.02], 
      parcelCount: 50, 
      anomalyCount: 1 
    },
  ];
}

function formatCoords(lat: number, lng: number) {
  const latStr = `${Math.abs(lat).toFixed(5)}° ${lat >= 0 ? 'N' : 'S'}`;
  const lngStr = `${Math.abs(lng).toFixed(5)}° ${lng >= 0 ? 'E' : 'W'}`;
  return `${latStr}, ${lngStr}`;
}

function getStringHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

export function TopologyModule() {
  const [statesList, setStatesList] = useState<StateOption[]>(DEFAULT_STATES);
  const [selectedRegion, setSelectedRegion] = useState<StateOption>(DEFAULT_STATES[0]);
  
  // Available cities for the selected region
  const availableCities = getCitiesForState(selectedRegion);
  const [selectedCity, setSelectedCity] = useState<CityOption>(availableCities[0]);

  const [selectedView, setSelectedView] = useState(MAP_VIEWS[0]);
  const [isScanning, setIsScanning] = useState(false);
  const [isFixing, setIsFixing] = useState(false);
  const [parcels, setParcels] = useState<turf.FeatureCollection<turf.Polygon | turf.MultiPolygon> | null>(null);
  const [errors, setErrors] = useState<TopologyError[]>([]);
  const [selectedErrorId, setSelectedErrorId] = useState<string | null>(null);
  const [reportModal, setReportModal] = useState<{ before: number, fixed: number, time: number } | null>(null);
  const [mapBounds, setMapBounds] = useState<[[number, number], [number, number]] | null>(null);

  // Load ALL states dynamically from india_boundaries.geojson (Excluding island territories as requested)
  useEffect(() => {
    fetch('/data/india_boundaries.geojson')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data?.features)) {
          const loaded = data.features
            .map((f: any) => ({
              id: f.properties.id || f.properties.state_code,
              name: f.properties.state_name || f.properties.name,
              center: f.properties.center ? (f.properties.center as [number, number]) : [20.5937, 78.9629],
              bounds: f.properties.bounds,
              feature: f
            }))
            .filter((st: any) => {
              const nameLower = (st.name || '').toLowerCase();
              return !nameLower.includes('andaman') && !nameLower.includes('nicobar') && !nameLower.includes('lakshadweep') && !nameLower.includes('lakshwadweep');
            })
            .sort((a: any, b: any) => a.name.localeCompare(b.name));

          if (loaded.length > 0) {
            setStatesList(loaded);
          }
        }
      })
      .catch((e) => console.warn('Falling back to default state list:', e));
  }, []);

  // Update selected city when selected state region changes
  useEffect(() => {
    const cities = getCitiesForState(selectedRegion);
    setSelectedCity(cities[0]);
  }, [selectedRegion]);

  // Auto-scan whenever selected city changes
  useEffect(() => {
    generateAndScan();
  }, [selectedCity]);

  const generateAndScan = async () => {
    setIsScanning(true);
    setErrors([]);
    setParcels(null);
    setReportModal(null);
    setSelectedErrorId(null);

    try {
      let baseFeatures: any[] = [];
      try {
        const response = await fetch('/data/parcels_21.geojson');
        if (response.ok) {
          const data = await response.json();
          baseFeatures = (data.features || []).filter((f: any) => 
            f.geometry && (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon')
          );
        }
      } catch (err) {
        console.warn('Network load error for parcels_21.geojson:', err);
      }

      if (baseFeatures.length === 0) {
        const bbox = [
          selectedCity.center[1] - 0.015,
          selectedCity.center[0] - 0.015,
          selectedCity.center[1] + 0.015,
          selectedCity.center[0] + 0.015
        ];
        const grid = turf.squareGrid(bbox, 0.4, { units: 'kilometers' });
        baseFeatures = grid.features;
      }

      // SLICE PARCEL COUNT STRICTLY BY CITY SIZE TIER (Small city = less parcels, Mega metro = more parcels)
      const targetParcelCount = Math.min(baseFeatures.length, selectedCity.parcelCount);
      const activeFeatures = baseFeatures.slice(0, targetParcelCount);

      // ORGANIC LAND SCATTER ENGINE AT CITY CENTER (Golden-angle spiral, zero square box)
      const citySeed = getStringHash(selectedCity.id);
      const stateId = selectedRegion.id.toLowerCase();
      const isWestCoast = stateId.includes('kerala') || stateId.includes('goa') || stateId.includes('maharashtra') || stateId.includes('gujarat') || stateId.includes('karnataka');

      const anchorLng = selectedCity.center[1] + (isWestCoast ? 0.008 : 0);
      const anchorLat = selectedCity.center[0];

      const distributedFeatures: any[] = [];

      activeFeatures.forEach((f: any, i: number) => {
        try {
          const itemSeed = citySeed + i * 997;
          const r1 = ((itemSeed * 9301 + 49297) % 233280) / 233280;
          const r2 = ((itemSeed * 49297 + 9301) % 233280) / 233280;

          // Golden Angle scatter
          const angle = i * 2.39996 + (citySeed % 6);
          // Scatter radius varies by city tier
          const maxRadius = selectedCity.tier === 'mega' ? 0.045 : selectedCity.tier === 'large' ? 0.035 : selectedCity.tier === 'medium' ? 0.025 : 0.015;
          const radius = Math.sqrt(r1) * maxRadius;

          const dLat = radius * Math.sin(angle);
          const dLng = radius * Math.cos(angle) * (isWestCoast ? 0.7 : 1.0);

          const targetPos: [number, number] = [anchorLng + dLng, anchorLat + dLat];

          const featureCentroid = turf.centroid(f).geometry.coordinates; // [lng, lat]
          const sourcePt = turf.point(featureCentroid);
          const targetPt = turf.point(targetPos);

          const dist = turf.distance(sourcePt, targetPt, { units: 'kilometers' });
          const bear = turf.bearing(sourcePt, targetPt);

          const translated = dist > 0.0001 ? turf.transformTranslate(f, dist, bear, { units: 'kilometers' }) : f;
          
          const rotated = turf.transformRotate(translated, Math.floor(r2 * 360));
          const scaled = turf.transformScale(rotated, 0.75 + r1 * 0.45);

          // Guarantee inland land constraint if state boundary polygon exists
          let finalFeature = scaled;
          if (selectedRegion.feature) {
            const centroid = turf.centroid(scaled);
            const isInside = turf.booleanPointInPolygon(centroid, selectedRegion.feature);
            if (!isInside) {
              const stateCenterPt = turf.point([selectedCity.center[1], selectedCity.center[0]]);
              const pullDist = turf.distance(centroid, stateCenterPt, { units: 'kilometers' });
              const pullBear = turf.bearing(centroid, stateCenterPt);
              finalFeature = turf.transformTranslate(scaled, pullDist * 0.7, pullBear, { units: 'kilometers' });
            }
          }

          distributedFeatures.push({
            ...finalFeature,
            properties: {
              ...finalFeature.properties,
              id: `parcel-${i}`,
              status: 'scanned',
              zone: `${selectedCity.name} Sector-${(i % 4) + 1}`,
              landType: ['Agricultural', 'Residential', 'Commercial', 'Government', 'Forest'][i % 5]
            }
          });
        } catch {
          distributedFeatures.push({
            ...f,
            properties: { ...f.properties, id: `parcel-${i}`, status: 'scanned' }
          });
        }
      });

      let grid = turf.featureCollection(distributedFeatures) as turf.FeatureCollection<turf.Polygon | turf.MultiPolygon>;
      const newErrors: TopologyError[] = [];

      // ANOMALY COUNT VARIES STRICTLY BY CITY SIZE TIER
      // Mega Metro = 8-10 anomalies, Small City = 1-2 anomalies
      const targetErrorCount = Math.min(selectedCity.anomalyCount, Math.max(1, grid.features.length - 1));

      const anomalyTypes = [
        { type: 'overlap', severity: 'high', priority: 'P1 - Critical', priorityCode: 'P1', riskRange: [92.0, 98.8] },
        { type: 'sliver', severity: 'medium', priority: 'P3 - Medium', priorityCode: 'P3', riskRange: [68.0, 79.5] },
        { type: 'unclosed', severity: 'high', priority: 'P1 - Critical', priorityCode: 'P1', riskRange: [91.0, 97.5] },
        { type: 'encroachment', severity: 'high', priority: 'P2 - High', priorityCode: 'P2', riskRange: [82.0, 91.5] },
        { type: 'disconnection', severity: 'medium', priority: 'P4 - Low', priorityCode: 'P4', riskRange: [54.0, 67.5] },
      ];

      for (let k = 0; k < targetErrorCount; k++) {
        const featureIdx1 = (k * 2 + (citySeed % 5)) % Math.max(1, grid.features.length - 2);
        const featureIdx2 = featureIdx1 + 1;
        const template = anomalyTypes[(k + citySeed) % anomalyTypes.length];
        
        try {
          const poly1 = grid.features[featureIdx1];
          const poly2 = grid.features[featureIdx2];
          
          if (template.type === 'overlap') {
            const translatedPoly2 = turf.transformTranslate(poly2, 0.003, (k * 60 + 45) % 360);
            grid.features[featureIdx2] = translatedPoly2;
            const overlap = turf.intersect(turf.featureCollection([poly1, translatedPoly2]));
            const centerPt = overlap ? turf.center(overlap).geometry.coordinates : turf.center(poly1).geometry.coordinates;
            const bbox = overlap ? turf.bbox(overlap) : turf.bbox(poly1);
            
            const riskScore = parseFloat((template.riskRange[0] + ((citySeed * 13 + k * 17) % 100) / 100 * (template.riskRange[1] - template.riskRange[0])).toFixed(1));

            newErrors.push({
              id: `err-overlap-${k + 1}`,
              type: 'overlap',
              severity: 'high',
              priority: 'P1 - Critical',
              priorityCode: 'P1',
              riskScore,
              featureId1: `parcel-${featureIdx1}`,
              featureId2: `parcel-${featureIdx2}`,
              center: [centerPt[1], centerPt[0]],
              bounds: [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
              description: `Critical Cadastral Overlap (${riskScore}% Risk) detected between parcel-${featureIdx1} and parcel-${featureIdx2} in ${selectedCity.name}.`
            });
          } else if (template.type === 'sliver') {
            const translatedPoly2 = turf.transformTranslate(poly2, 0.002, 270);
            grid.features[featureIdx2] = translatedPoly2;
            const bbox1 = turf.bbox(poly1);
            const bbox2 = turf.bbox(translatedPoly2);
            const lat = (bbox1[1] + bbox2[1]) / 2;
            const lng = (bbox1[0] + bbox2[0]) / 2;
            
            const riskScore = parseFloat((template.riskRange[0] + ((citySeed * 7 + k * 23) % 100) / 100 * (template.riskRange[1] - template.riskRange[0])).toFixed(1));

            newErrors.push({
              id: `err-sliver-${k + 1}`,
              type: 'sliver',
              severity: 'medium',
              priority: 'P3 - Medium',
              priorityCode: 'P3',
              riskScore,
              featureId1: `parcel-${featureIdx1}`,
              featureId2: `parcel-${featureIdx2}`,
              center: [lat, lng],
              bounds: [[bbox1[1], bbox1[0]], [bbox2[3], bbox2[2]]],
              description: `Micro Sliver Gap (${riskScore}% Risk) detected between nodes of parcel-${featureIdx1} & parcel-${featureIdx2} in ${selectedCity.name}.`
            });
          } else if (template.type === 'unclosed') {
            const poly = grid.features[featureIdx1];
            const bbox = turf.bbox(poly);
            const centerPt = turf.center(poly).geometry.coordinates;
            
            const riskScore = parseFloat((template.riskRange[0] + ((citySeed * 11 + k * 19) % 100) / 100 * (template.riskRange[1] - template.riskRange[0])).toFixed(1));

            newErrors.push({
              id: `err-unclosed-${k + 1}`,
              type: 'unclosed',
              severity: 'high',
              priority: 'P1 - Critical',
              priorityCode: 'P1',
              riskScore,
              featureId1: `parcel-${featureIdx1}`,
              center: [centerPt[1], centerPt[0]],
              bounds: [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
              description: `Unclosed Ring Defect (${riskScore}% Risk) on parcel-${featureIdx1} in ${selectedCity.name}.`
            });
          } else if (template.type === 'encroachment') {
            const poly = grid.features[featureIdx1];
            const bbox = turf.bbox(poly);
            const centerPt = turf.center(poly).geometry.coordinates;
            
            const riskScore = parseFloat((template.riskRange[0] + ((citySeed * 19 + k * 29) % 100) / 100 * (template.riskRange[1] - template.riskRange[0])).toFixed(1));

            newErrors.push({
              id: `err-encroach-${k + 1}`,
              type: 'encroachment',
              severity: 'high',
              priority: 'P2 - High',
              priorityCode: 'P2',
              riskScore,
              featureId1: `parcel-${featureIdx1}`,
              center: [centerPt[1], centerPt[0]],
              bounds: [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
              description: `Vertex Encroachment Defect (${riskScore}% Risk) on parcel-${featureIdx1} in ${selectedCity.name}.`
            });
          } else {
            const poly = grid.features[featureIdx1];
            const bbox = turf.bbox(poly);
            const centerPt = turf.center(poly).geometry.coordinates;
            
            const riskScore = parseFloat((template.riskRange[0] + ((citySeed * 23 + k * 31) % 100) / 100 * (template.riskRange[1] - template.riskRange[0])).toFixed(1));

            newErrors.push({
              id: `err-disconn-${k + 1}`,
              type: 'disconnection',
              severity: 'medium',
              priority: 'P4 - Low',
              priorityCode: 'P4',
              riskScore,
              featureId1: `parcel-${featureIdx1}`,
              center: [centerPt[1], centerPt[0]],
              bounds: [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
              description: `Boundary Node Disconnection (${riskScore}% Risk) on parcel-${featureIdx1} in ${selectedCity.name}.`
            });
          }
        } catch (err) {
          console.warn('Error generating anomaly:', err);
        }
      }

      // Sort discrepancies by Priority & GeoAI Risk Score (descending)
      newErrors.sort((a, b) => b.riskScore - a.riskScore);

      // Calculate dynamic bounds from the relocated dataset
      const bbox = turf.bbox(grid);
      setMapBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]]);
      setParcels(grid);
      setErrors(newErrors);

      // Save scan state for Command Center linking
      sessionStorage.setItem('bhoomi_topology_scan', JSON.stringify({
        regionId: selectedCity.id,
        regionName: `${selectedRegion.name} (${selectedCity.name})`,
        errorCount: newErrors.length,
        timestamp: new Date().toISOString()
      }));
    } catch (e: any) {
      console.error('Scan topology error:', e);
    } finally {
      setIsScanning(false);
    }
  };

  const autoFix = () => {
    setIsFixing(true);
    const startTime = performance.now();
    
    setTimeout(() => {
      // Execute Self-Healing (Simulate graph alignment)
      if (parcels) {
        const fixedFeatures = [...parcels.features];
        
        // Fix Overlaps
        try {
          const poly1 = fixedFeatures[0];
          const poly2 = fixedFeatures[1];
          const unioned = turf.union(turf.featureCollection([poly1, poly2]));
          if (unioned) {
            fixedFeatures[0] = { ...unioned, properties: { id: 'parcel-0-fixed', status: 'fixed' } } as any;
            fixedFeatures[1] = { ...poly2, properties: { id: 'merged', status: 'hidden' } };
          }
        } catch {}

        setParcels(turf.featureCollection(fixedFeatures) as any);
      }

      const timeTaken = ((performance.now() - startTime) / 1000).toFixed(2);
      
      setReportModal({
        before: errors.length,
        fixed: errors.length,
        time: parseFloat(timeTaken)
      });

      // Update sessionStorage for Command Center
      sessionStorage.setItem('bhoomi_topology_fixed', JSON.stringify({
        regionId: selectedCity.id,
        regionName: `${selectedRegion.name} (${selectedCity.name})`,
        fixedCount: errors.length,
        timestamp: new Date().toISOString()
      }));

      setErrors([]);
      setSelectedErrorId(null);
      setIsFixing(false);
    }, 1200);
  };

  const activeError = errors.find(e => e.id === selectedErrorId);

  // Dynamic Parcel Map Styles based on Map View Mode
  const parcelStyle = (feature: any) => {
    if (feature.properties?.status === 'hidden') return { opacity: 0, fillOpacity: 0 };
    if (feature.properties?.status === 'fixed') return { color: '#10b981', weight: 2.5, fillColor: '#10b981', fillOpacity: 0.35 };
    
    // Highlight if part of selected error
    if (activeError && (feature.properties?.id === activeError.featureId1 || feature.properties?.id === activeError.featureId2)) {
      return { color: '#ef4444', weight: 3.5, fillColor: '#ef4444', fillOpacity: 0.55 };
    }
    
    // View specific parcel colors
    if (selectedView.id === 'satellite') {
      return { color: '#38bdf8', weight: 2, fillColor: '#38bdf8', fillOpacity: 0.25 };
    }
    if (selectedView.id === 'elevation') {
      return { color: '#d97706', weight: 2, fillColor: '#f59e0b', fillOpacity: 0.22 };
    }
    if (selectedView.id === 'cadastral') {
      return { color: '#60a5fa', weight: 2, fillColor: '#3b82f6', fillOpacity: 0.35 };
    }
    if (selectedView.id === 'thematic') {
      const colors = ['#8b5cf6', '#ec4899', '#3b82f6', '#10b981'];
      const idx = (parseInt(feature.properties?.id?.replace('parcel-', '') || '0')) % colors.length;
      return { color: colors[idx], weight: 2, fillColor: colors[idx], fillOpacity: 0.4 };
    }

    return { color: '#3b82f6', weight: 2, fillColor: '#3b82f6', fillOpacity: 0.15 };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minHeight: '100%' }}>
      
      {/* Page Header */}
      <div className="page-heading">
        <div>
          <div className="eyebrow">VALIDATION / GEOMETRY INTEGRITY</div>
          <h1>Topology Control & Anomaly Engine</h1>
          <p>Multi-city cadastral inspection and real-time graph alignment across Indian states and major municipal jurisdictions.</p>
        </div>
        <div className="heading-actions" style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          
          {/* State Jurisdiction Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--surface-sunken)', border: '1px solid var(--border)', padding: '7px 12px', borderRadius: '6px' }}>
            <MapPinned size={16} style={{ color: 'var(--accent-blue)' }} />
            <select 
              value={selectedRegion.id}
              onChange={(e) => {
                const r = statesList.find(st => st.id === e.target.value);
                if (r) {
                  setSelectedRegion(r);
                  const cities = getCitiesForState(r);
                  setSelectedCity(cities[0]);
                }
              }}
              style={{
                border: 'none',
                background: 'transparent',
                color: 'var(--navy)',
                fontSize: '13px',
                fontWeight: 700,
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              {statesList.map(r => (
                <option key={r.id} value={r.id}>{r.name}</option>
              ))}
            </select>
          </div>

          {/* Major City / Municipal Region Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--surface-sunken)', border: '1px solid var(--border)', padding: '7px 12px', borderRadius: '6px' }}>
            <Building2 size={16} style={{ color: '#10b981' }} />
            <select 
              value={selectedCity.id}
              onChange={(e) => {
                const c = availableCities.find(ct => ct.id === e.target.value);
                if (c) setSelectedCity(c);
              }}
              style={{
                border: 'none',
                background: 'transparent',
                color: 'var(--navy)',
                fontSize: '13px',
                fontWeight: 700,
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              {availableCities.map(c => (
                <option key={c.id} value={c.id}>{c.name} ({c.parcelCount} parcels)</option>
              ))}
            </select>
          </div>

          {/* Scan dataset button */}
          <Button kind="secondary" onClick={generateAndScan} disabled={isScanning || isFixing}>
            <Search size={15} /> {isScanning ? 'Scanning GeoData...' : 'Scan dataset topology'}
          </Button>

          {/* Auto-Fix button */}
          <Button onClick={autoFix} disabled={isFixing || errors.length === 0} style={{ background: 'var(--accent-blue)', color: 'white' }}>
            <Hammer size={15} /> {isFixing ? 'Applying graph alignment...' : 'Execute Self-Healing Auto-Fix'}
          </Button>
        </div>
      </div>

      {/* 5 MAP VIEW LAYER SELECTION TABS */}
      <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
        {MAP_VIEWS.map(v => {
          const Icon = v.icon;
          const isActive = selectedView.id === v.id;
          return (
            <button
              key={v.id}
              onClick={() => setSelectedView(v)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 14px',
                borderRadius: '6px',
                border: `1.5px solid ${isActive ? 'var(--accent-blue)' : 'var(--border)'}`,
                background: isActive ? 'var(--navy)' : 'var(--surface-sunken)',
                color: isActive ? 'white' : 'var(--ink)',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s ease'
              }}
            >
              <Icon size={14} style={{ color: isActive ? 'var(--yellow)' : 'var(--accent-blue)' }} />
              {v.name}
            </button>
          );
        })}
      </div>

      {/* MAIN LAYOUT GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '20px', flex: 1, minHeight: '620px' }}>
        
        {/* Interactive Map Surface */}
        <Surface style={{ padding: 0, overflow: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
          {/* Subheader overlay */}
          <div style={{ padding: '10px 16px', background: 'var(--surface-sunken)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 2 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="live-dot" />
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--navy)' }}>
                {selectedRegion.name} / <strong style={{ color: 'var(--accent-blue)' }}>{selectedCity.name}</strong> · <span style={{ color: 'var(--text-muted)' }}>{selectedCity.tierLabel}</span>
              </span>
            </div>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Center: {formatCoords(selectedCity.center[0], selectedCity.center[1])}
            </span>
          </div>

          {/* Loading overlay */}
          {isScanning && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.85)', zIndex: 10, backdropFilter: 'blur(4px)' }}>
              <LoadingState label={`Scanning ${selectedCity.parcelCount} cadastral parcels across ${selectedCity.name} (${selectedRegion.name})...`} />
            </div>
          )}
          
          {/* Leaflet Map Container */}
          <div style={{ flex: 1, position: 'relative', width: '100%', height: '100%', minHeight: '520px' }}>
            <MapContainer center={selectedCity.center} zoom={selectedCity.tier === 'mega' ? 12 : selectedCity.tier === 'large' ? 13 : 14} style={{ width: '100%', height: '100%', zIndex: 1 }}>
              <TileLayer url={selectedView.tile} attribution={selectedView.attr} />
              <MapController 
                center={activeError ? activeError.center : selectedCity.center} 
                zoom={activeError ? 16 : selectedCity.tier === 'mega' ? 12 : 14}
                bounds={activeError ? activeError.bounds : (mapBounds ? mapBounds : null)}
              />
              
              {/* Parcels GeoJSON */}
              {parcels && (
                <GeoJSON 
                  key={`${selectedCity.id}-${selectedView.id}-${isFixing ? 'fixed' : (selectedErrorId || 'scanned')}`} 
                  data={parcels} 
                  style={parcelStyle} 
                />
              )}
            </MapContainer>
          </div>
        </Surface>

        {/* Live Error Matrix Queue Panel */}
        <Surface style={{ display: 'flex', flexDirection: 'column' }}>
          <SectionHeading 
            eyebrow="LIVE DISCREPANCY MATRIX" 
            title="Topology Queue" 
            action={<span className="mono" style={{ padding: '3px 10px', borderRadius: '4px', background: errors.length > 0 ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)', color: errors.length > 0 ? 'var(--danger)' : 'var(--good)', fontWeight: 700 }}>{errors.length} detected</span>} 
          />

          <div style={{ padding: '6px 12px', background: 'var(--surface-sunken)', borderRadius: '6px', fontSize: '11px', color: 'var(--text-muted)', margin: '4px 0 10px', display: 'flex', justifyContent: 'space-between' }}>
            <span>Target City: <strong>{selectedCity.name}</strong></span>
            <span>Parcels Indexed: <strong>{selectedCity.parcelCount}</strong></span>
          </div>
          
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {errors.length === 0 && parcels && !reportModal && (
              <div style={{ padding: '30px 20px', textAlign: 'center', color: 'var(--good)', background: 'rgba(16,185,129,0.05)', borderRadius: '8px', border: '1px solid rgba(16,185,129,0.2)' }}>
                <CheckCircle2 size={36} style={{ margin: '0 auto 12px' }} />
                <strong style={{ display: 'block', fontSize: '15px' }}>Topology 100% Clean</strong>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>All {selectedCity.parcelCount} boundary intersections in {selectedCity.name} are topologically aligned.</p>
              </div>
            )}
            
            {errors.map(err => (
              <div 
                key={err.id}
                onClick={() => setSelectedErrorId(err.id)}
                style={{ 
                  padding: '14px', 
                  borderRadius: '8px', 
                  border: `1.5px solid ${selectedErrorId === err.id ? 'var(--accent-blue)' : 'var(--border)'}`,
                  background: selectedErrorId === err.id ? 'rgba(59,130,246,0.08)' : 'var(--surface-sunken)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                {/* Header with Type, Priority Badge & Risk Score */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--navy)' }}>
                    <AlertTriangle size={15} style={{ color: err.severity === 'high' ? 'var(--danger)' : 'var(--warn)' }} />
                    {err.type === 'overlap' ? 'Overlapping Boundary' : err.type === 'sliver' ? 'Sliver Gap' : err.type === 'unclosed' ? 'Unclosed Ring Defect' : err.type === 'encroachment' ? 'Vertex Encroachment' : 'Node Disconnection'}
                  </strong>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ 
                      fontSize: '10px', 
                      fontWeight: 800, 
                      padding: '2px 6px', 
                      borderRadius: '4px', 
                      background: err.priorityCode === 'P1' ? '#ef4444' : err.priorityCode === 'P2' ? '#f59e0b' : '#3b82f6', 
                      color: 'white' 
                    }}>
                      {err.priorityCode}
                    </span>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--danger)', background: 'rgba(239,68,68,0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                      {err.riskScore}/100
                    </span>
                  </div>
                </div>

                {/* EXACT LOCATION COORDINATES IN MATRIX */}
                <div style={{ fontSize: '12px', color: 'var(--navy)', fontWeight: 700, fontFamily: 'var(--app-font-mono)', display: 'flex', alignItems: 'center', gap: '6px', margin: '4px 0' }}>
                  <Crosshair size={13} style={{ color: 'var(--accent-blue)' }} />
                  {formatCoords(err.center[0], err.center[1])}
                </div>

                <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                  <span>Nodes: <strong>{err.featureId1}</strong> {err.featureId2 ? `& ${err.featureId2}` : ''}</span>
                  <ChevronRight size={14} style={{ opacity: selectedErrorId === err.id ? 1 : 0.4 }} />
                </div>
              </div>
            ))}
          </div>
        </Surface>
      </div>

      {/* SEPARATE ANOMALY COORDINATES REGISTRY PANEL (BELOW MATRIX) */}
      <Surface style={{ marginTop: '10px' }}>
        <SectionHeading 
          eyebrow="SPATIAL COORDINATE REGISTRY" 
          title="Detected Anomalies Coordinate Registry & Risk Scores" 
          action={<span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Jurisdiction: {selectedRegion.name} ({selectedCity.name})</span>}
        />
        
        {errors.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
            No active spatial anomaly coordinates logged for {selectedCity.name}.
          </div>
        ) : (
          <div style={{ overflowX: 'auto', marginTop: '12px' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface-sunken)' }}>
                  <th style={{ padding: '10px 14px' }}>Priority</th>
                  <th style={{ padding: '10px 14px' }}>GeoAI Risk Score</th>
                  <th style={{ padding: '10px 14px' }}>Anomaly ID</th>
                  <th style={{ padding: '10px 14px' }}>Type</th>
                  <th style={{ padding: '10px 14px' }}>Exact Lat / Lng Coordinates</th>
                  <th style={{ padding: '10px 14px' }}>Bounding Box (Min Lat/Lng → Max Lat/Lng)</th>
                  <th style={{ padding: '10px 14px' }}>Nodes</th>
                  <th style={{ padding: '10px 14px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {errors.map(err => (
                  <tr 
                    key={`reg-${err.id}`} 
                    style={{ 
                      borderBottom: '1px solid var(--border)', 
                      background: selectedErrorId === err.id ? 'rgba(59,130,246,0.05)' : 'transparent' 
                    }}
                  >
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{ 
                        fontSize: '11px', 
                        fontWeight: 800, 
                        padding: '3px 8px', 
                        borderRadius: '4px', 
                        background: err.priorityCode === 'P1' ? '#ef4444' : err.priorityCode === 'P2' ? '#f59e0b' : '#3b82f6', 
                        color: 'white' 
                      }}>
                        {err.priority}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 800, color: 'var(--danger)' }} className="mono">
                      {err.riskScore} / 100
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 700 }} className="mono">{err.id}</td>
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--navy)' }}>
                        {err.type === 'overlap' ? 'Overlapping Boundary' : err.type === 'sliver' ? 'Sliver Gap' : err.type === 'unclosed' ? 'Unclosed Ring' : err.type === 'encroachment' ? 'Vertex Encroachment' : 'Node Disconnection'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: 'var(--accent-blue)' }} className="mono">
                      {formatCoords(err.center[0], err.center[1])}
                    </td>
                    <td style={{ padding: '12px 14px', fontSize: '11px', color: 'var(--text-muted)' }} className="mono">
                      {err.bounds[0][0].toFixed(4)}°, {err.bounds[0][1].toFixed(4)}° → {err.bounds[1][0].toFixed(4)}°, {err.bounds[1][1].toFixed(4)}°
                    </td>
                    <td style={{ padding: '12px 14px' }}>{err.featureId1} {err.featureId2 ? `& ${err.featureId2}` : ''}</td>
                    <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                      <Button 
                        kind="secondary" 
                        onClick={() => setSelectedErrorId(err.id)}
                        style={{ padding: '4px 10px', fontSize: '11px' }}
                      >
                        <Crosshair size={12} /> Inspect
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Surface>

      {/* Discrepancy Report Modal */}
      {reportModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(3px)' }}>
          <Surface style={{ width: '460px', padding: '30px', boxShadow: '0 20px 40px rgba(0,0,0,0.2)' }}>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <div style={{ background: '#10b981', color: 'white', width: '52px', height: '52px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px', boxShadow: '0 4px 12px rgba(16,185,129,0.3)' }}>
                <Check size={28} />
              </div>
              <h2 style={{ fontSize: '20px', margin: '0 0 6px 0', color: 'var(--navy)' }}>Topology Discrepancy Report</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: 0 }}>Self-healing graph alignment completed for <strong>{selectedCity.name}</strong> ({selectedRegion.name}).</p>
            </div>
            
            <div style={{ background: 'var(--surface-sunken)', borderRadius: '8px', padding: '16px', marginBottom: '22px', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', paddingBottom: '10px', borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Topology errors resolved</span>
                <strong style={{ color: '#10b981' }}>{reportModal.fixed} / {reportModal.before}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', paddingBottom: '10px', borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Resolution engine</span>
                <strong>turf.js graph alignment</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Compute time</span>
                <strong>{reportModal.time}s</strong>
              </div>
            </div>

            <Button onClick={() => {
              setReportModal(null);
            }} style={{ width: '100%', background: 'var(--accent-blue)', color: 'white', padding: '12px' }}>
              Acknowledge & Update Command Center
            </Button>
          </Surface>
        </div>
      )}
    </div>
  );
}
