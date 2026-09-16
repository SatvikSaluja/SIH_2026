import { FeatureCollection, Feature, Polygon } from 'geojson';

export interface ParcelProperties {
  ulpin: string;
  area_sqm: number;
  ownership_status: 'Private' | 'Government' | 'Disputed';
  confidence_score: number;
  owner_name: string;
}

const FIRST_NAMES = ['Aarav', 'Vihaan', 'Aditya', 'Sai', 'Arjun', 'Diya', 'Ananya', 'Aadhya', 'Priya', 'Kavya'];
const LAST_NAMES = ['Sharma', 'Patel', 'Singh', 'Kumar', 'Das', 'Gupta', 'Verma', 'Reddy', 'Rao', 'Nair'];

export function generateMicroParcels(
  bounds: [[number, number], [number, number]],
  regionCode: string,
  count: number = 200
): FeatureCollection<Polygon, ParcelProperties> {
  const features: Feature<Polygon, ParcelProperties>[] = [];

  const [latMin, lngMin] = bounds[0];
  const [latMax, lngMax] = bounds[1];

  // Restrict to a smaller central area of the bounds for a denser grid effect
  const centerLat = (latMin + latMax) / 2;
  const centerLng = (lngMin + lngMax) / 2;
  
  const spanLat = (latMax - latMin) * 0.1; // 10% of total span
  const spanLng = (lngMax - lngMin) * 0.1;
  
  const innerLatMin = centerLat - spanLat / 2;
  const innerLngMin = centerLng - spanLng / 2;

  const parcelSizeDeg = 0.0008;

  for (let i = 0; i < count; i++) {
    const lat = innerLatMin + Math.random() * spanLat;
    const lng = innerLngMin + Math.random() * spanLng;

    const w = parcelSizeDeg * (0.5 + Math.random() * 1.5);
    const h = parcelSizeDeg * (0.5 + Math.random() * 1.5);

    const coordinates = [
      [
        [lng, lat],
        [lng + w, lat],
        [lng + w, lat + h],
        [lng, lat + h],
        [lng, lat], 
      ]
    ];

    const idNum = Math.floor(1000 + Math.random() * 9000);
    const subNum = Math.floor(100 + Math.random() * 900);
    const ulpin = `IN-${regionCode}-${idNum}-P${subNum}`;
    
    const area = Math.floor(w * h * 111000 * 111000 * Math.cos(lat * Math.PI / 180));
    
    const rand = Math.random();
    const status = rand > 0.85 ? 'Disputed' : rand > 0.65 ? 'Government' : 'Private';
    const confidence = 85 + (Math.random() * 14.9);
    
    const ownerName = status === 'Government' 
      ? 'State Govt. / Municipal'
      : `${FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)]} ${LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)]}`;

    features.push({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates,
      },
      properties: {
        ulpin,
        area_sqm: area,
        ownership_status: status as any,
        confidence_score: Number(confidence.toFixed(1)),
        owner_name: ownerName
      }
    });
  }

  return {
    type: 'FeatureCollection',
    features,
  };
}
