# MedVisionAI: Unified Clinical Intelligence Platform

An open-source clinical decision-support platform combining **Brain Tumor MRI** (deep ensemble classification, U-Net lesion segmentation, and radiomic risk scoring) and **Colorectal Histopathology** (Vision Transformer classification and token Grad-CAM explainability) within an August Health EHR-styled clinical interface.

---

## Features & Clinical Modalities

### 1. Brain Tumor MRI Studio (`BRISC 2025` Cohort)
- **Multi-Class Classification**: Identifies Glioma, Meningioma, Pituitary Adenoma, and Healthy Brain scans.
- **Deep Model Consensus**: Live comparison between a 5-fold EfficientNet-B0 ensemble (99.30% test accuracy), single EfficientNet-B0, ConvNeXt-Tiny, and ResNet-18 backbones.
- **Lesion Segmentation**: 256×256 U-Net (`seg_unet_small.pt`) with horizontal-flip Test-Time Augmentation (TTA), Magma spatial probability heatmaps, and lesion contour boundary extraction.
- **Classical Radiomics**: Scikit-learn pipeline evaluating Random Forest, Gradient Boosting, SVM, Decision Tree, and Logistic Regression on 7 radiomic features (Shannon entropy, blur score, Canny edge density, mean intensity, standard deviation, image quality, and anatomical plane code).
- **Clinical Risk Score (0–100)**: Composite risk triage score calculating urgency priority (Routine, Specialist Review, Urgent Board Review).

### 2. Colorectal Histopathology Studio (`NCT-CRC-HE-100K` / `CRC-VAL-HE-7K`)
- **Foundation Vision Transformer**: Powered by Lunit DINO ViT-S/8 and Owkin Phikon ViT-B/16.
- **9-Class Tissue Classification**: Classifies colorectal adenocarcinoma epithelium (TUM), normal mucosa (NORM), cancer-associated stroma (STR), lymphocytes (LYM), debris/necrosis (DEB), mucus (MUC), smooth muscle (MUS), adipose tissue (ADI), and background (BACK).
- **Token Grad-CAM**: Computes attention gradients directly from patch tokens entering the final transformer LayerNorm block.
- **Mosaic Whole-Slide Scanner**: Partitions multi-patch whole-slide images into 224×224 px tiles, filters glass background, and computes tumour tile fraction.
- **Domain Verification**: Validates H&E staining color profiles before inference to prevent out-of-distribution errors.

### 3. Medical AI Assistant & Specialist Triage
- **Clinical Scope Guard**: Medical-only conversational assistant rejecting non-clinical queries.
- **Case-Grounded Synthesis**: Automatically digests complete scan telemetry (multi-model consensus, segmented lesion burden %, and classical ML predictions) into structured consultation notes.
- **Triage Queue**: Automated priority queueing across Neuro-Oncology, Neuroradiology, and Gastrointestinal Pathology specialists.

---

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server, endpoints, and triage logic
│   └── ml_engine.py         # PyTorch inference engine (ViT, CNNs, UNet, Classical ML)
├── frontend/
│   ├── index.html           # Single-page clinical workbench interface
│   ├── styles.css           # August Health EHR warm-clinical design system
│   ├── app.js               # Reactive client application logic
│   └── favicon.svg          # Medical cross clinical SVG icon
├── braintumor/
│   ├── models/              # Checkpoints (.pt, .joblib)
│   ├── sample_mris/         # Curated demonstration MRIs
│   └── submissionwithnomodels/  # Research graphs and validation reports
├── medvisionai/
│   ├── models/              # Lunit ViT weights and head checkpoints
│   └── examples/            # Sample H&E patches and mosaic scans
├── run_app.sh               # Portable startup script
├── Dockerfile               # Production container definition
├── requirements.txt         # Frozen Python dependencies
└── README.md
```

---

## Getting Started

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/medvisionai.git
   cd medvisionai
   ```

2. **Create and activate a Python 3.11 environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Start the platform:**
   ```bash
   ./run_app.sh
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### Docker Deployment

```bash
docker build -t medvisionai .
docker run -p 8000:8000 medvisionai
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET`, `HEAD` | Serves the web clinical interface |
| `/api/health` | `GET`, `HEAD` | Platform health check and engine metadata |
| `/api/samples` | `GET` | List curated sample MRIs and histology patches |
| `/api/benchmarks` | `GET` | Research metrics, comparison tables, and figures |
| `/api/analyze/brain` | `POST` | Multi-backbone CNN, U-Net, and classical ML analysis |
| `/api/analyze/colorectal` | `POST` | ViT 9-class prediction, Grad-CAM, and mosaic tile scan |
| `/api/chat` | `POST` | Medical copilot consultation and case synthesis |
| `/api/appointments` | `GET`, `POST` | Specialist referral requests and coordinator queue |
| `/api/appointments/{id}` | `PATCH` | Update triage status or scheduled clinic times |

---

## Disclaimer

MedVisionAI is an academic and research decision-support prototype. It is **not** an approved medical device under FDA or CE regulations. It must not be used as the sole basis for clinical diagnosis or treatment planning.
