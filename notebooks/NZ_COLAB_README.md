# New Zealand Colab pilot

1. Open https://colab.research.google.com/ and use File → Upload notebook to open `GeoCadastra_NZ_Colab.ipynb`.
2. Run the upload cell and select `artifacts/nz_colab_bundle.zip` from this workspace.
3. Mount Google Drive for resumable checkpoints. Choose a free GPU runtime before the training cells.
4. Run the one-epoch benchmark, then continue training. Evaluate the test location after choosing the model.

The ZIP contains prepared data; you do not need to clone or publish this repository, create a LINZ account, or rent a GPU. Free Colab GPU availability is not guaranteed. If unavailable, wait or use the script on your own GPU. Keep Colab's installed CUDA PyTorch: do not install the project's backend requirements in this notebook.

## Files and provenance

`data/nz_pilot/manifest.json` lists the six selected source tiles, immutable hashes of downloaded images, hashes of prepared tensors and parcel JSON, spatial splits, coordinate system, source URLs and limitations. The imagery collection and STAC items retain capture dates and provider attribution. Native GeoTIFFs are retained locally in each selected tile directory but omitted from the upload ZIP to save bandwidth.

Imagery: Christchurch 2025, 7.5 cm native RGB, resampled with averaging to 30 cm (1200 × 800) for a cheap pilot. CRS is NZTM2000 / EPSG:2193. Source: https://nz-imagery.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2025_0.075m/rgb/2193/collection.json

Imagery is CC BY 4.0, produced by Aerial Surveys, licensed by Environment Canterbury, and hosted/processed by Toitū Te Whenua LINZ. Preserve collection attribution when sharing.

Parcel source: public LINZ-derived ArcGIS mirror, explicitly recorded in the manifest. It is not a direct authenticated LINZ export. The service metadata is saved; the map source is https://data.linz.govt.nz/layer/50772-nz-primary-parcels/ . Survey and calculated areas are kept separately in CSV; missing values stay missing. These CSVs are not used as training inputs. No owner names are requested. For authoritative evaluation, replace the mirror snapshot with a verified LINZ extract and review source licence/accuracy metadata.

## What is trained

A small convolutional RGB encoder/decoder predicts a parcel-map boundary band. Only boundary labels are trained; no invented road/building classes, terrain or uncertainty targets. GroupNorm allows small batches. CUDA uses mixed precision. Validation loss selects best.pt and controls early stopping. last.pt includes optimizer, scaler and random state for epoch-boundary resume. Dataset fingerprints and configuration prevent accidentally resuming against changed labels.

This is a separate baseline, **not the application's MultiTaskNet**. Do not point GEOCADASTRA_MODEL_WEIGHTS at best.pt. The production model expects RGB+nDSM and different outputs. Integration and comparison against that model remain subsequent work.

Train/validation/test use different source tiles with at least 100 m separation and no shared intersecting parcel IDs. All crops from one tile stay in its split. The tiny sample supports an engineering experiment, not a national performance estimate. Crops must be at least 75% valid. The outer raster edge and missing parcel coverage are masked. Reference lines are taken from original polygon boundaries, not boundaries created by clipping polygons to a tile.

Reported precision, recall, F1 and IoU measure pixel overlap against a three-pixel-wide reference boundary band (30 cm pixels), not independently surveyed positional accuracy. Probability outputs are not calibrated confidence. The model may not learn useful boundaries from this tiny sample; use validation metrics and preview, not training loss alone.

Current parcel geometry may differ from March 2025 imagery. No DSM/DTM, historical input parcel map, independent survey checks or per-parcel accuracy tolerance is included. These are not silently manufactured. Inspect overview.jpg; its red lines are parcel references, not predictions. The study tiles are rectangular sampling windows, not full reconstruction blocks. Boundary fragments and partial parcels are acceptable for pixel training but not full-parcel area evaluation. records.csv flags fully contained parcels.

## Local commands

Run from repository root with Python, torch, and scripts/colab/requirements.txt installed:

```bash
python scripts/colab/download_nz.py
python scripts/colab/package_nz.py
python scripts/colab/train_nz.py --epochs 1 --out runs/nz_rgb
python scripts/colab/train_nz.py --epochs 20 --out runs/nz_rgb --resume
python scripts/colab/train_nz.py --out runs/nz_rgb --evaluate
```

The time limit is checked between epochs; an epoch may exceed it. Checkpoints are atomic local-file replacements; persistent Drive storage improves recovery but cannot guarantee survival of a remote filesystem interruption. The uncompleted epoch repeats after a disconnect. No hard paid-service spending controls are configured because this notebook does not provision paid services.
