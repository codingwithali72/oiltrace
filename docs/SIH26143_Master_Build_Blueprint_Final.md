# SIH26143 / OilTrace — MASTER Build Blueprint (Final, Consolidated)
### Everything merged: your build plan + mentor's ecosystem diagram + continuous AIS monitoring

---

## PHASE 0 — What Was Kept, Changed, and Rejected (read this first)

**Kept from your original build blueprint, unchanged:** the full detection → drift → attribution pipeline, all dataset/model sources, the scoring formula, UI style spec, testing strategy. This remains the backbone.

**Kept from your mentor's ecosystem diagram, now built as real phases (not just future vision), since you have real prep time:**
- Real alerting (email/SMS) — Phase 16
- Cloud storage for raw scenes/outputs — Phase 15
- A feedback/retraining loop — Phase 19
- A richer, more complete data schema covering AIS, satellite, environmental, geospatial, and event data — Phase 5 (as one database, see below)

**Rejected from the mentor's diagram, with reasons, regardless of available time:**
- **Five separate databases** (AIS DB, Satellite DB, Environmental DB, Geospatial DB, Event DB) → kept as **one PostgreSQL+PostGIS database with five well-structured table groups** instead. Splitting into five databases adds real operational overhead (five connections, cross-database joins) with zero benefit a single well-designed schema doesn't already give you — more prep time makes this schema *better*, not a reason to fragment it.
- **Live regulatory integration (EMSA/IMO reporting)** → kept as a documented, working REST API endpoint that *could* feed such a system, honestly labeled "designed for future regulatory integration," since real integration requires institutional access no student project has.

**Elevated per your sir's instruction, and correctly framed:** continuous AIS anomaly monitoring is now **Phase 6**, a real always-on parallel track, not an optional afterthought. The framing stays exactly as established earlier and doesn't change: **this narrows where you look next — it never detects a spill by itself.** AIS has no signal that oil exists in the water; only SAR (or a report) can confirm that. Your sir's underlying point — that Sentinel-1's ~6-day single-satellite revisit means you can't rely on scanning alone — is correct, and continuous AIS-anomaly watching is a legitimate, real-world way (used operationally as "tip-off tasking") to decide where to prioritize the next available SAR pass.

---

## PHASE 1 — Regional Scope: What Gets Trained Globally vs. What's Region-Specific

This distinction matters and should be explicit in your documentation and PPT:

| Component | Training data scope | Why |
|---|---|---|
| Oil detection model (DeepLabV3+) | **Global** — Krestenitis Zenodo dataset (worldwide SAR scenes) | The visual pattern of "oil dampens capillary waves" is universal physics — it doesn't matter what ocean the training images came from |
| Ship detection model (Faster R-CNN) | **Global** — xView3-SAR dataset (worldwide scenes) | Same reasoning — a ship looks like a bright radar point regardless of location |
| Look-alike classifier (LightGBM) | **Global** — Krestenitis's own look-alike labels | Biogenic slicks/low-wind physics are also universal |
| Drift simulation (OpenDrift) | **Regional** — INCOIS/HYCOM current + ERA5 wind data for the **Arabian Sea / Kerala coast** specifically | Ocean currents and wind patterns are location-specific — you must use real regional data here, not generic/global averages, or your drift results will be physically wrong for your demo region |
| AIS data (both case-based and continuous monitoring) | **Regional** — Arabian Sea shipping lanes, Kochi port approach corridor | Your demo scenario and continuous-monitoring pitch are both explicitly India-focused (NTRO, Kerala coast) — scope your synthetic AIS generator and monitoring logic to real regional shipping lane geometry, not generic open ocean |
| Demo/case scenario | **Regional** — one specific spill scenario near Kochi/Arabian Sea (real archive scene or curated synthetic) | Matches your actual target use case and lets you reference the real MSC Elsa 3 context credibly |

**In one sentence for your PPT:** the AI models are trained on global data because the physics they learn is universal, but the drift simulation and AIS monitoring are configured specifically for Indian waters because that's where they need to be physically accurate.

---

## PHASE 2 — Accounts and Tools (same as before, plus cloud storage)

| # | What | Where | Notes |
|---|---|---|---|
| 1-13 | *(All accounts from the original build blueprint: Python, Node, Java/Maven, Docker, AI coding assistant, GitHub, Colab, Kaggle, Copernicus Data Space, NASA Earthdata, CDS/ERA5, QGIS, Postman)* | — | Unchanged — see original list |
| 14 | AWS free tier (S3) OR Google Cloud free tier (GCS) | aws.amazon.com/free / cloud.google.com/free | For Phase 15's cloud storage — pick whichever your team is more comfortable with |
| 15 | SendGrid (email) or Twilio (SMS) free tier | sendgrid.com / twilio.com | For Phase 16's real alerting |

---

## PHASE 3 — Get Every Dataset (unchanged from your original document)

Same exact sources as before — Zenodo Krestenitis dataset, Copernicus Data Space / ASF Vertex for real Sentinel-1 scenes, xView3-SAR (download only 3–5 scenes), marinecadastre.gov + Global Fishing Watch for AIS format reference, HYCOM/Copernicus Marine for currents, ERA5 via CDS for wind, GEBCO for bathymetry. No changes here — this part of your original plan was already correct.

---

## PHASE 4 — Get Every Pretrained Model (unchanged)

Same as before: `segmentation-models-pytorch` auto-downloads ImageNet weights for DeepLabV3+; `torchvision`'s Faster R-CNN auto-downloads COCO weights; check Hugging Face for any existing SAR-specific checkpoints as a possible shortcut; `pip install opendrift` for the physics engine.

---

## PHASE 5 — Single Consolidated Database Schema (replaces the mentor's 5-database split)

One PostgreSQL + PostGIS database, structured into five logical table groups matching what the mentor's diagram wanted — without the overhead of separate databases:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- Group 1: AIS trajectory data (live + historical + synthetic)
CREATE TABLE ais_ping (
    id BIGSERIAL PRIMARY KEY,
    mmsi TEXT NOT NULL,
    position GEOMETRY(Point, 4326) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    speed_knots DOUBLE PRECISION,
    course DOUBLE PRECISION,
    vessel_type TEXT,
    is_synthetic BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_ais_position ON ais_ping USING GIST (position);
CREATE INDEX idx_ais_mmsi_time ON ais_ping (mmsi, timestamp);

-- Group 2: Satellite image metadata (not the raw imagery itself — that lives in cloud storage, Phase 15)
CREATE TABLE sar_scene (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    acquisition_time TIMESTAMPTZ NOT NULL,
    footprint GEOMETRY(Polygon, 4326),
    storage_url TEXT NOT NULL,  -- points to the file in your cloud bucket
    source TEXT  -- 'sentinel-1' or 'eos-04'
);

-- Group 3: Environmental data cache (avoids re-fetching HYCOM/ERA5 repeatedly)
CREATE TABLE environmental_snapshot (
    id BIGSERIAL PRIMARY KEY,
    region GEOMETRY(Polygon, 4326),
    timestamp TIMESTAMPTZ,
    wind_speed_ms DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    current_data_url TEXT  -- pointer to cached NetCDF in cloud storage
);

-- Group 4: Geospatial reference data (static)
CREATE TABLE geospatial_zone (
    id SERIAL PRIMARY KEY,
    zone_type TEXT,  -- 'port', 'shipping_lane', 'marine_protected_area', 'offshore_platform'
    name TEXT,
    geom GEOMETRY(Geometry, 4326)
);

-- Group 5: Oil spill case/event data (the core working table)
CREATE TABLE spill_case (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sar_scene_id UUID REFERENCES sar_scene(id),
    detected_at TIMESTAMPTZ,
    spill_geom GEOMETRY(Polygon, 4326),
    area_km2 DOUBLE PRECISION,
    age_band_label TEXT,
    age_confidence DOUBLE PRECISION,
    origin_polygon GEOMETRY(Polygon, 4326),
    forecast_polygon GEOMETRY(Polygon, 4326),
    status TEXT DEFAULT 'processing'
);

CREATE TABLE vessel_suspect (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spill_case_id UUID REFERENCES spill_case(id),
    mmsi TEXT,
    track_geom GEOMETRY(LineString, 4326),
    proximity_score DOUBLE PRECISION,
    iou_match_score DOUBLE PRECISION,
    ais_gap_score DOUBLE PRECISION,
    dark_vessel_flag BOOLEAN DEFAULT FALSE,
    final_score DOUBLE PRECISION
);

-- Continuous monitoring watchlist (Phase 6 output — separate from case-based suspects)
CREATE TABLE ais_watchlist (
    id BIGSERIAL PRIMARY KEY,
    mmsi TEXT,
    flag_type TEXT,  -- 'ais_gap', 'loitering', 'erratic_course'
    location GEOMETRY(Point, 4326),
    flagged_at TIMESTAMPTZ,
    reviewed BOOLEAN DEFAULT FALSE
);
```

This single schema gives you every category the mentor's diagram wanted, with normal foreign-key relationships and one connection to manage.

---

## PHASE 6 — Continuous AIS Anomaly Monitoring (elevated, per your sir's instruction)

**Purpose, stated plainly again since it must never be misrepresented:** this is a background process that continuously scans AIS traffic (all vessel types, all the time) for suspicious behavior, and writes flagged vessels to `ais_watchlist`. It runs independent of any specific spill case. **It does not detect oil spills.** It exists to answer "given that we can't scan the whole ocean with SAR constantly, where should we prioritize looking next?"

### Implementation
1. Write a standalone Python script (`ais_monitor.py`) using the same gap/loitering/course-deviation detection logic you'll build in Phase 12, but running it across your **entire** regional AIS dataset (real sample + synthetic), not scoped to one spill's origin window.
2. Run it on a schedule — a simple `cron` job, or a loop with `time.sleep()` for the demo, or a proper task scheduler (APScheduler in Python) if you want it production-shaped.
3. Any vessel tripping a flag (AIS gap > threshold, speed dropping to near-zero for an extended period near a shipping lane = loitering, or a sudden sharp course change) gets written to the `ais_watchlist` table with its flag type, location, and timestamp.
4. Build a small, separate "Watchlist" screen on the frontend — a simple table, clearly labeled **"Regions Prioritized for Next Satellite Pass"** — never "Detected Spills."
5. **For your demo:** run this monitor across your synthetic AIS dataset (Phase 12) and show that your planted "suspect" vessel's AIS gap appears on the Watchlist *before* you even open the SAR-based case for that spill — this is your live illustration of the concept, and it directly answers your sir's brief without overclaiming what AIS can do.

### How to describe this correctly to a judge (unchanged from before, repeated because it matters)
"A continuous AIS-behavior monitor that prioritizes where we request or prioritize the next satellite pass — functioning like a Coast Guard patrol tip-off in the real world. It complements SAR-based detection; it does not replace it, since AIS alone can never confirm that oil is present."

---

## PHASE 7 — Train the Oil Detection Model in Colab (full cell-by-cell detail)

Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**.

**Cell 1 — install everything:**
```python
!pip install segmentation-models-pytorch rasterio albumentations
```

**Cell 2 — mount Drive and get the dataset (do this once, dataset persists in Drive after):**
```python
from google.colab import drive
drive.mount('/content/drive')
# Upload the Krestenitis Zenodo zip to your Drive first, then:
!unzip -q "/content/drive/MyDrive/krestenitis_dataset.zip" -d /content/data
```

**Cell 3 — Dataset class:**
```python
import torch, numpy as np, rasterio
from torch.utils.data import Dataset
import albumentations as A

class SARSpillDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.transform = A.Compose([
            A.RandomCrop(256, 256), A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        ]) if augment else None

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            img = src.read(1).astype(np.float32)
        img = 10 * np.log10(np.clip(img, 1e-6, None))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = np.array(rasterio.open(self.mask_paths[idx]).read(1))
        binary_mask = (mask == 1).astype(np.float32)  # class 1 = oil, per Krestenitis labeling
        if self.transform:
            aug = self.transform(image=img, mask=binary_mask)
            img, binary_mask = aug["image"], aug["mask"]
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(binary_mask).unsqueeze(0).float()
```

**Cell 4 — build file lists and split train/val (80/20):**
```python
import glob, random
images = sorted(glob.glob("/content/data/images/*.tif"))
masks  = sorted(glob.glob("/content/data/masks/*.png"))
combined = list(zip(images, masks)); random.seed(42); random.shuffle(combined)
split = int(0.8 * len(combined))
train_pairs, val_pairs = combined[:split], combined[split:]
```

**Cell 5 — model, training loop (as in the earlier architecture document, unchanged):**
```python
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

train_ds = SARSpillDataset(*zip(*train_pairs))
val_ds = SARSpillDataset(*zip(*val_pairs), augment=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet",
                           in_channels=1, classes=1).to("cuda")
loss_fn = smp.losses.DiceLoss(mode="binary")
bce_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def evaluate(model, loader):
    model.eval(); intersection, union = 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to("cuda"), masks.to("cuda")
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            intersection += (preds * masks).sum().item()
            union += ((preds + masks) > 0).float().sum().item()
    return intersection / (union + 1e-8)

best_iou = 0
for epoch in range(20):
    model.train()
    for imgs, masks in train_loader:
        imgs, masks = imgs.to("cuda"), masks.to("cuda")
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks) + bce_fn(logits, masks)
        loss.backward(); optimizer.step()
    val_iou = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: val IoU = {val_iou:.3f}")
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model, "/content/drive/MyDrive/spill_segmentation_best.pt")
print(f"Done. Best IoU: {best_iou:.3f}")
```

**Cell 6 — download the weights to your local machine** (or just keep them in Drive and mount Drive later from your `ml-service`).

**Sanity check before moving on:** load one validation image, run it through the model, and visually plot the prediction next to the ground truth mask in the same Colab notebook (`matplotlib.pyplot.imshow`) — confirm by eye that it's actually learning the right shape before trusting the IoU number alone.

---

## PHASE 8 — Pixel-to-Coordinate Mapping (unchanged from original document)

Use `rasterio.transform.xy()` on GRD scenes (already geocoded) to convert mask boundary pixels to real lat/lon, then build a `shapely.Polygon` — verify visually in QGIS before proceeding. (Full detail already covered in your original build blueprint — no changes here.)

---

## PHASE 9 — Geometry, Age, and Look-Alike Classifier (training detail added)

Geometry and age estimation are unchanged from your original document. For the **look-alike classifier training**, do this in the same Colab notebook as Phase 7 (reuse the loaded dataset):

```python
from skimage.feature import graycomatrix, graycoprops
import lightgbm as lgb
import numpy as np

def extract_features(image_patch, wind_speed):
    glcm = graycomatrix((image_patch * 255).astype(np.uint8), distances=[1],
                         angles=[0], levels=256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        wind_speed,
    ]

# Build X (features) and y (1=oil, 0=look-alike) from Krestenitis's own labeled
# look-alike regions (class 2 in most versions of this dataset) vs oil regions (class 1)
X_train, y_train = [], []  # populate by looping through labeled patches as above

clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)

import joblib
joblib.dump(clf, "/content/drive/MyDrive/lookalike_classifier.pkl")
```

---

## PHASE 10 — Train the Ship Detection Model (xView3, cell-by-cell)

**Cell 1:**
```python
!pip install torchvision
```

**Cell 2 — load a pretrained Faster R-CNN and prepare for fine-tuning:**
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2  # background + ship
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to("cuda")
```

**Cell 3 — chip your 3–5 downloaded xView3 scenes into 800×800 tiles** (reuse chipping code from a DIUx-xView baseline repo, per your Phase 3 notes — this step is dataset-preprocessing-specific and worth copying from a working reference rather than writing blind).

**Cell 4 — standard torchvision object-detection training loop** (fine-tune for 5–10 epochs on your small chipped subset — a full from-scratch treatment isn't needed here since this is deliberately a lightweight fine-tune, as noted in your original document).

**Cell 5 — save weights**, same pattern as Phase 7.

---

## PHASE 11 — Drift Engine Setup (unchanged, region-specific per Phase 1 above)

Same as your original document: `pip install opendrift`, test with bundled demo data first, then swap in real **Arabian Sea-region** HYCOM current + ERA5 wind NetCDF files for both the case-based hindcast and the demo forecast.

---

## PHASE 12 — Case-Based AIS Engine + Synthetic Data Generator (unchanged, region-specific)

Same track-reconstruction and gap-detection logic as your original document, scoped to the **Kochi/Arabian Sea shipping corridor** for your synthetic generator (realistic 10–20 knot cargo speeds along the real approach lane to Kochi port) — this is also the exact logic reused by Phase 6's continuous monitor, just run in two different modes (case-scoped vs. always-on).

---

## PHASE 13 — Vessel Suspicion Scoring Formula (unchanged)

Same composite formula as your original document:
```
final_score = (0.25 × proximity_score) + (0.30 × forward_match_score)
            + (0.20 × ais_gap_score) + (0.15 × dark_vessel_flag)
            + (0.10 × vessel_type_suitability_score)
```

---

## PHASE 14 — Backend (Spring Boot, wired to the Phase 5 schema)

Same as your original document — `start.spring.io` with Web, Data JPA, PostgreSQL, Security dependencies. Now with entity classes matching all six tables from Phase 5 (`AisPing`, `SarScene`, `EnvironmentalSnapshot`, `GeospatialZone`, `SpillCase`, `VesselSuspect`, `AisWatchlist`). Test every endpoint in Postman before frontend work.

---

## PHASE 15 — Cloud Storage (NEW — real phase, not just future vision)

1. Create an S3 bucket (AWS free tier) or GCS bucket (GCP free tier).
2. When a new SAR scene is processed, upload the raw `.tif` and any generated overlay images to this bucket; store only the resulting URL in the `sar_scene.storage_url` column (Phase 5) — never store large binary files directly in PostgreSQL.
3. Same pattern for cached environmental NetCDF files (`environmental_snapshot.current_data_url`), so you don't re-download HYCOM/ERA5 data every time you reprocess a region.
4. Java's AWS SDK (`software.amazon.awssdk`) or Google Cloud's Java client library plugs directly into your Spring Boot backend for upload/download calls.

---

## PHASE 16 — Real Alerts (NEW — real phase)

1. Sign up for SendGrid (email) or Twilio (SMS) free tier.
2. In Spring Boot, add a notification service that fires when a new `spill_case` is created with a high-confidence top suspect (e.g., final_score > 0.7): send an email/SMS to a configured investigator address with the case ID and a link to the dashboard.
3. This is a small, genuinely working feature now that you have real time — don't overbuild it (no need for a full notification preferences system), just a working trigger-and-send.

---

## PHASE 17 — Frontend UI/UX (unchanged design spec, same investigation-dashboard style: dark theme, monospace data readouts, map-dominant layout, sharp corners — full detail in your original document)

---

## PHASE 18 — Report Generation (unchanged from original document: OpenPDF/PDFBox in Spring Boot, `/api/cases/{id}/report` endpoint, "Download Report" button)

---

## PHASE 19 — Feedback & Continuous Learning Loop (NEW — real phase)

1. Add a simple "Confirm/Reject" button on the Suspect Evidence Panel — when an investigator marks a suspect as confirmed or cleared, write that outcome to a new `feedback` table (`spill_case_id`, `mmsi`, `confirmed BOOLEAN`, `notes TEXT`).
2. Periodically (manually is fine for a demo — no need for automation), export confirmed/cleared cases and use them to **adjust your composite scoring weights** (Phase 13) — e.g., if dark-vessel flags are consistently confirmed correct, that weight could increase.
3. This doesn't need to be a full automated ML retraining pipeline to be a legitimate, working feature — a documented feedback table plus a manual weight-review process is honest, real, and demonstrates the concept without overbuilding.

---

## PHASE 20 — Testing and Debugging (unchanged from original document: pytest, JUnit, QGIS visual verification, integration test end-to-end, common bug checklist)

---

## PHASE 21 — Working With Your AI Coding Assistant (unchanged: one phase at a time, review everything, small commits)

---

## PHASE 22 — Deployment (unchanged: Docker Compose locally first, then free-tier host, test from a fresh browser before the actual demo)

---

## Summary: What's Genuinely New in This Merged Version vs. Your Original Document

- Explicit regional-scope table (Phase 1) — answers exactly which data trains globally vs. which is Kerala/Arabian-Sea-specific
- Continuous AIS anomaly monitoring elevated to its own real phase (Phase 6), correctly framed
- Full cell-by-cell Colab training code for both models and the look-alike classifier (Phases 7, 9, 10)
- One consolidated database schema absorbing all five of the mentor's data categories (Phase 5)
- Three genuinely new working features enabled by having real prep time: cloud storage (15), real alerts (16), feedback loop (19)
- Two things explicitly kept rejected regardless of time: five separate databases, and fake regulatory integration
