-- Run after 0001 with the application's schema first in search_path.
-- Old faces remain unassigned: historical identity cannot safely be guessed.
BEGIN;
ALTER TABLE recorded_parcels ADD COLUMN IF NOT EXISTS area_tolerance_m2 double precision NOT NULL DEFAULT 0.01;
ALTER TABLE faces ADD COLUMN IF NOT EXISTS recorded_parcel_id bigint REFERENCES recorded_parcels(id);
CREATE INDEX IF NOT EXISTS ix_faces_recorded_parcel_id ON faces(recorded_parcel_id);
ALTER TABLE face_boundaries ADD COLUMN IF NOT EXISTS ring integer NOT NULL DEFAULT 0;
ALTER TABLE face_boundaries DROP CONSTRAINT IF EXISTS face_boundaries_pkey;
ALTER TABLE face_boundaries ADD PRIMARY KEY (face_id, ring, position);
COMMIT;
