# Final Google Colab workflow

Open GeoCadastra_Stage4_GPU.ipynb in Google Colab, select a GPU runtime, and run its cells in order. Upload artifacts/stage4_gpu_bundle.zip when prompted. Rebuild that archive locally with `python scripts/colab/package_stage4.py` after source changes. This workflow does not require a private GitHub token or a pushed commit.

The notebook retains Colab's CUDA torch/torchvision installation, installs only training dependencies, mounts Drive before training, and writes every completed epoch to Drive. It uses 64 synthetic training wards, 16 separate validation wards, batch size 16 and a 400-epoch total cosine schedule. Geometry uses 64×64 pixels at 1 m/pixel, minor spacing 30, wall rendering width 1.5 m; task loss weights are SDF 2, other tasks 0.5. These last settings differ from the plain TrainConfig defaults in the pasted recipe.

Run the one-epoch benchmark, then the continuation cell. After a disconnect rerun setup and continuation with the same source and configuration. The checkpoint is atomic on the local filesystem; Drive synchronization is still managed by Colab/Drive, so confirm files are present before intentionally ending a session. Lowering batch size after an out-of-memory error requires a fresh run folder; the resume configuration check deliberately rejects schedule/dataset changes.

`last.pt` is the resumable state. `last.pt.best` contains the epoch with the lowest validation loss. Evaluation loads the latter and measures ten new synthetic wards with seeds starting at 100000. Read heldout_metrics.json, including false-positive diagnostics, before choosing weights. The Stage 4 pytest test retrains its own model and does not evaluate these checkpoints. Its old one-sided distance metric alone cannot establish model quality.

The standalone evaluator uses synthetic criteria plus a full-image-prediction guard; it is not certification and does not claim exact equivalence to the pytest fixture. Do not repeatedly tune on this test set and continue calling it independent evaluation; reserve new areas/seeds for final assessment.

This trains the actual MultiTaskNet, not the separate RGB-only NZ baseline. The 100 PNG/height examples are not 100 labelled parcel training samples. Real imagery still needs co-located parcel labels, appropriate CRS handling and spatially separate train/validation/test areas. GPU availability and run time vary. No GPU run or accuracy success is claimed from local CPU smoke tests.
