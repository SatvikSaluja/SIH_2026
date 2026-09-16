-- Apply to the application's schema with psql -v ON_ERROR_STOP=1.
-- The caller's search_path must name that schema first, followed by public.
BEGIN;
ALTER TABLE ward_jobs ADD COLUMN IF NOT EXISTS coreg_transform jsonb;
ALTER TABLE legacy_records ADD COLUMN IF NOT EXISTS original_geom geometry(GEOMETRY,32643);
UPDATE legacy_records SET original_geom = geom WHERE original_geom IS NULL;
CREATE INDEX IF NOT EXISTS idx_legacy_records_original_geom ON legacy_records USING gist(original_geom);

ALTER TABLE survey_points ADD COLUMN IF NOT EXISTS purpose varchar NOT NULL DEFAULT 'fusion';
-- Old points may already have influenced geometry. Never relabel them as held out.
CREATE INDEX IF NOT EXISTS ix_survey_points_purpose ON survey_points(purpose);
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'recorded_parcels'::regclass AND conname = 'uq_recorded_parcel_ward_id') THEN
        ALTER TABLE recorded_parcels ADD CONSTRAINT uq_recorded_parcel_ward_id UNIQUE (ward_job_id, id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'survey_points'::regclass AND conname = 'ck_survey_purpose') THEN
        ALTER TABLE survey_points ADD CONSTRAINT ck_survey_purpose CHECK (purpose IN ('fusion','calibration','evaluation'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'survey_points'::regclass AND conname = 'fk_survey_parcel_ward') THEN
        ALTER TABLE survey_points ADD CONSTRAINT fk_survey_parcel_ward FOREIGN KEY (ward_job_id, parcel_id) REFERENCES recorded_parcels(ward_job_id, id);
    END IF;
END $$;

ALTER TABLE conflicts ADD COLUMN IF NOT EXISTS kind varchar NOT NULL DEFAULT 'source_disagreement';
ALTER TABLE conflicts ADD COLUMN IF NOT EXISTS detail jsonb NOT NULL DEFAULT '{}';
ALTER TABLE conflicts ALTER COLUMN disagreement_m DROP NOT NULL;
CREATE TABLE IF NOT EXISTS coregistrations (
    id bigserial PRIMARY KEY,
    ward_job_id bigint NOT NULL REFERENCES ward_jobs(id),
    changeset_id bigint NOT NULL REFERENCES changesets(id),
    transform jsonb NOT NULL,
    control_points jsonb NOT NULL,
    residual_m double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_coregistrations_ward_job_id ON coregistrations(ward_job_id);
COMMIT;
