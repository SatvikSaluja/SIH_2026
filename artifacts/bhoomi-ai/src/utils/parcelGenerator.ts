// Was: generateMicroParcels(), a fake-data generator producing random
// polygon geometry, random ULPIN-formatted ids, random Disputed/Private/
// Government status, a random confidence score, and randomly generated
// Indian person names as fake property owners -- synthetic land-ownership
// records with real-sounding names attached, none of it derived from any
// real parcel. Removed; ParcelExplorerModal now fetches real parcels from
// the geocadastra-backed /api/parcels endpoint (see bhoomi.ts). Fields
// geocadastra has no real concept for (ownership_status, confidence_score,
// owner_name) are nullable here rather than fabricated.
export interface ParcelProperties {
  ulpin: string;
  area_sqm: number;
  ownership_status: string | null;
  confidence_score: number | null;
  owner_name: string | null;
}
