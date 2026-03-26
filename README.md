# License Plate Recognition

Real-time license plate detection and OCR from video using OpenCV + TensorFlow.

---

## Project Structure

```
license_plate_recognition/
├── plate_finder.py      # Plate detection & character segmentation
├── ocr.py               # TensorFlow OCR classifier
├── main.py              # Video runner (CLI)
├── train_ocr.py         # Model training script
├── requirements.txt
└── model/               # (you create this)
    ├── binary_128_0.50_ver3.pb
    └── binary_128_0.50_labels_ver2.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic training data & train the model
```bash
python train_ocr.py --generate --samples 500 --epochs 25
```
This will:
- Render 500 images per character (36 chars × 500 = 18 000 images)
- Train a CNN classifier
- Save `model/binary_128_0.50_ver3.pb` and `model/binary_128_0.50_labels_ver2.txt`

### 3. Run on a video
```bash
python main.py --video test.MOV
```

### 4. Run on webcam
```bash
python main.py --video 0
```

Press **Q** in any OpenCV window to quit.

---

## Getting Real Training Data

| Dataset | URL | Notes |
|---|---|---|
| Kaggle Car Plates | kaggle.com/datasets/andrewmvd/car-plate-detection | Easy start |
| CCPD | github.com/detectRecog/CCPD | 250k+ images |
| OpenALPR Benchmarks | github.com/openalpr/benchmarks | US/EU plates |
| Chars74K | ee.surrey.ac.uk/CVSSP/demos/chars74k | A–Z, 0–9 chars |

After downloading, arrange as:
```
training_data/
  0/  1/  2/ ... A/  B/ ... Z/
```
Then run:
```bash
python train_ocr.py --data-dir training_data/ --epochs 30
```

---

## CLI Options

### main.py
| Flag | Default | Description |
|---|---|---|
| `--video` | `test.MOV` | Video path or `0` for webcam |
| `--model` | `model/binary_128_0.50_ver3.pb` | Frozen model path |
| `--labels` | `model/binary_128_0.50_labels_ver2.txt` | Labels file |
| `--min-area` | `4100` | Min plate area (px²) |
| `--max-area` | `15000` | Max plate area (px²) |

### train_ocr.py
| Flag | Default | Description |
|---|---|---|
| `--generate` | off | Generate synthetic images first |
| `--data-dir` | `training_data` | Dataset root folder |
| `--samples` | `500` | Images per character |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--model-out` | `model/…ver3.pb` | Output model path |
| `--labels-out` | `model/…ver2.txt` | Output labels path |

---

## Notes

- The detector expects **8-character plates** (tweak `check_plate` in `plate_finder.py` if your region differs).
- For best accuracy use real cropped character images from actual plates.
- The frozen `.pb` format is TF1-compatible; loaded via `tf.compat.v1`.