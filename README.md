# EfficientNet-B3 Branch (`efficientnetb3-model`)

Standalone EfficientNet-B3 vegetable classifier API for Google Cloud Run.

This repository branch contains **only** `api_efficientnet_b3/` — no POS UI, no YOLO code, no datasets.

## Quick start

1. Upload model to GCS (not Git):

```bash
gsutil cp efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
```

2. Deploy:

```bash
gcloud builds submit --config=api_efficientnet_b3/cloudbuild.yaml .
```

3. Open Swagger UI: `https://<cloud-run-url>/docs`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/infer` | Upload image (form field: `file`) |

## Model files

`.onnx` and `.pt` files are **never** committed to Git. They live in GCS and are downloaded during Cloud Build.

See `api_efficientnet_b3/README.md` and `api_efficientnet_b3/models/README.md`.

## Postman

Import `api_efficientnet_b3/VeggieLens_API.postman_collection.json`.
