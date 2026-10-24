"""Unified MedVisionAI Backend API."""
from __future__ import annotations

import csv
import io
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .ml_engine import (
    BRAIN_CLASS_LABELS,
    BRAIN_DATA_DIR,
    BRAIN_GRAPHS_DIR,
    BRAIN_MODELS_DIR,
    BRAIN_REPORTS_DIR,
    BRAIN_SAMPLES_DIR,
    CRC_CLASS_NAMES,
    CRC_EXAMPLES_DIR,
    CRC_MODELS_DIR,
    DEVICE,
    ROOT_DIR,
    BrainTumorEngine,
    ColorectalEngine,
)

FRONTEND_DIR = ROOT_DIR / "frontend"
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 25 * 1024 * 1024))  # 25MB safety limit
ALLOWED_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(
    title="MedVisionAI Clinical Intelligence Platform",
    description="Unified AI Diagnostics & Triage for Colorectal Histopathology and Brain Tumor MRI.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/crc-samples", StaticFiles(directory=str(CRC_EXAMPLES_DIR)), name="crc_samples")
app.mount("/static/brain-samples", StaticFiles(directory=str(BRAIN_SAMPLES_DIR)), name="brain_samples")
app.mount("/static/brain-graphs", StaticFiles(directory=str(BRAIN_GRAPHS_DIR)), name="brain_graphs")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

crc_engine = ColorectalEngine()
brain_engine = BrainTumorEngine()

SPECIALISTS = [
    {
        "id": "neuro-oncology",
        "badge": "NO",
        "title": "Consultant Neuro-Oncologist",
        "department": "Neuro-Oncology & Neurosurgery",
        "modality": "brain_mri",
        "focus": "Glioma, meningioma & pituitary adenoma treatment planning, surgical resection & stereotactic radiosurgery",
        "hospital": "August Clinical Neuroscience Center",
    },
    {
        "id": "neuroradiology",
        "badge": "NR",
        "title": "Senior Neuroradiologist & MRI AI Specialist",
        "department": "Diagnostic Neuroradiology",
        "modality": "brain_mri",
        "focus": "Multi-planar brain MRI interpretation, volumetric tumor segmentation & saliency verification",
        "hospital": "August Clinical Neuroscience Center",
    },
    {
        "id": "gi-pathology",
        "badge": "GI",
        "title": "Consultant Gastrointestinal Pathologist",
        "department": "Anatomical Pathology",
        "modality": "colorectal",
        "focus": "Colorectal adenocarcinoma, H&E tissue morphology, cancer-associated stroma & biopsy review",
        "hospital": "August University Hospital",
    },
    {
        "id": "digital-pathology",
        "badge": "AI",
        "title": "Digital Pathology & Foundation Model Specialist",
        "department": "Computational Pathology",
        "modality": "colorectal",
        "focus": "ViT Grad-CAM heatmap interpretation, tile-scan tissue composition & model calibration",
        "hospital": "August University Hospital",
    },
    {
        "id": "colorectal-surgery",
        "badge": "CS",
        "title": "Consultant Colorectal Oncological Surgeon",
        "department": "Colorectal Surgery",
        "modality": "colorectal",
        "focus": "Surgical resection pathways and multidisciplinary tumour board referrals",
        "hospital": "August University Hospital",
    },
]

URGENCY_INFO = {
    "urgent": {
        "priority": 1,
        "label": "Urgent",
        "target": "Within 48 hours",
        "description": "Suspected high-grade malignancy (e.g. Glioma or Colorectal Adenocarcinoma) requiring rapid board review.",
    },
    "soon": {
        "priority": 2,
        "label": "Soon",
        "target": "Within 2 weeks",
        "description": "Abnormal lesion or moderate-confidence finding requiring specialist consultation.",
    },
    "routine": {
        "priority": 3,
        "label": "Routine",
        "target": "Within 6 weeks",
        "description": "Benign/normal tissue, second opinion, or routine follow-up.",
    },
}

APPOINTMENTS_STORE: list[dict[str, Any]] = [
    {
        "id": "req-neuro-001",
        "created_at": "2026-10-24T09:15:00Z",
        "updated_at": "2026-10-24T09:15:00Z",
        "organisation": "Central Radiology Lab",
        "requester_name": "Dr. Amine Mansouri",
        "contact": "a.mansouri@centralrad.org",
        "hospital": "August Clinical Neuroscience Center",
        "department": "Neuro-Oncology & Neurosurgery",
        "specialist": "Consultant Neuro-Oncologist",
        "specialist_id": "neuro-oncology",
        "urgency": "urgent",
        "priority": 1,
        "reason": "Axial T1 MRI indicates hyperintense mass consistent with glioma; segmentation burden 4.82% with elevated perilesional contrast.",
        "preferred": "Weekday mornings (Mon/Tue)",
        "patient_ref": "MRN-2026-8841",
        "case_summary": "Brain MRI Analysis (glioma_axial_t1.jpg): Predicted glioma (98.5% confidence, 5-fold EfficientNet-B0 ensemble); Segmented tumor area 4.82% (medium); Risk score 71.4/100 (high priority).",
        "status": "pending",
        "appointment_at": None,
        "location": "",
        "staff_note": "",
    },
    {
        "id": "req-crc-002",
        "created_at": "2026-10-24T10:30:00Z",
        "updated_at": "2026-10-24T10:30:00Z",
        "organisation": "University Histopathology Dept",
        "requester_name": "Dr. Sarah Benali",
        "contact": "s.benali@unipath.edu",
        "hospital": "August University Hospital",
        "department": "Anatomical Pathology",
        "specialist": "Consultant Gastrointestinal Pathologist",
        "specialist_id": "gi-pathology",
        "urgency": "urgent",
        "priority": 1,
        "reason": "Colonic biopsy H&E patch shows atypical glandular proliferation flagged as colorectal adenocarcinoma epithelium.",
        "preferred": "As soon as possible",
        "patient_ref": "PATH-2026-3109",
        "case_summary": "Colorectal H&E Analysis (test_colorectal_adenocarcinoma_epithelium_0.png): Tumour tissue detected; Predicted Colorectal adenocarcinoma epithelium (99.6% calibrated confidence).",
        "status": "pending",
        "appointment_at": None,
        "location": "",
        "staff_note": "",
    },
    {
        "id": "req-neuro-003",
        "created_at": "2026-10-23T14:20:00Z",
        "updated_at": "2026-10-23T16:45:00Z",
        "organisation": "Regional Diagnostic Imaging",
        "requester_name": "Dr. Karim Touati",
        "contact": "k.touati@regdiag.org",
        "hospital": "August Clinical Neuroscience Center",
        "department": "Diagnostic Neuroradiology",
        "specialist": "Senior Neuroradiologist & MRI AI Specialist",
        "specialist_id": "neuroradiology",
        "urgency": "soon",
        "priority": 2,
        "reason": "Coronal T1 scan shows well-circumscribed sellar/parasellar lesion consistent with pituitary adenoma; requesting volumetric review.",
        "preferred": "Wednesday afternoon",
        "patient_ref": "MRN-2026-7712",
        "case_summary": "Brain MRI Analysis (pituitary_coronal_sample.png): Predicted pituitary tumor (99.1% confidence); Segmented area 1.85% (medium); Risk score 48.2/100 (medium priority).",
        "status": "confirmed",
        "appointment_at": "2026-10-28T14:00",
        "location": "Neuroradiology Suite B, Floor 2",
        "staff_note": "Confirmed with Senior Neuroradiology team. Please bring prior DICOM series.",
    },
]


def _sort_appointments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            r["status"] != "pending",
            r.get("priority", 9),
            r.get("created_at", ""),
        ),
    )


def _with_queue_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = _sort_appointments(rows)
    pending_ids = [r["id"] for r in sorted_rows if r["status"] == "pending"]
    out = []
    for r in sorted_rows:
        item = dict(r)
        item["queue_position"] = (pending_ids.index(r["id"]) + 1) if r["id"] in pending_ids else None
        item["urgency_info"] = URGENCY_INFO.get(r["urgency"], URGENCY_INFO["routine"])
        out.append(item)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_study_benchmarks() -> dict[str, Any]:
    classical_rows = []
    for r in _read_csv(BRAIN_REPORTS_DIR / "model_comparison_table.csv"):
        classical_rows.append(
            {
                "model": r.get("model", "Unknown"),
                "accuracy": round(_safe_float(r.get("accuracy")), 4),
                "precision": round(_safe_float(r.get("precision")), 4),
                "recall": round(_safe_float(r.get("recall")), 4),
                "f1_score": round(_safe_float(r.get("f1_score")), 4),
                "auc_roc": round(_safe_float(r.get("auc_roc")), 4),
                "cv_mean_f1": round(_safe_float(r.get("cv_mean_f1")), 4),
                "training_time_s": round(_safe_float(r.get("training_time_seconds")), 3),
            }
        )
    classical_rows.sort(key=lambda x: x["f1_score"], reverse=True)

    cnn_backbones = []
    for r in _read_csv(BRAIN_REPORTS_DIR / "cnn_backbone_benchmark.csv"):
        cnn_backbones.append(
            {
                "backbone_name": r.get("backbone_name", ""),
                "run_name": r.get("run_name", ""),
                "best_epoch": int(_safe_float(r.get("best_epoch"))),
                "val_accuracy": round(_safe_float(r.get("val_accuracy")), 4),
                "val_macro_f1": round(_safe_float(r.get("val_macro_f1")), 4),
                "test_accuracy": round(_safe_float(r.get("test_accuracy")), 4),
                "test_macro_f1": round(_safe_float(r.get("test_macro_f1")), 4),
                "test_accuracy_tta": round(_safe_float(r.get("test_accuracy_tta")), 4),
                "test_macro_f1_tta": round(_safe_float(r.get("test_macro_f1_tta")), 4),
            }
        )

    cnn_per_class = []
    for r in _read_csv(BRAIN_REPORTS_DIR / "cnn_classification_report.csv"):
        lbl = r.get("") or r.get("label") or ""
        if lbl in BRAIN_CLASS_LABELS:
            cnn_per_class.append(
                {
                    "label": lbl,
                    "precision": round(_safe_float(r.get("precision")), 4),
                    "recall": round(_safe_float(r.get("recall")), 4),
                    "f1_score": round(_safe_float(r.get("f1-score")), 4),
                    "support": int(_safe_float(r.get("support"))),
                }
            )

    seg_models = []
    for r in _read_csv(BRAIN_REPORTS_DIR / "segmentation_model_benchmark.csv"):
        seg_models.append(
            {
                "model_name": r.get("model_name", ""),
                "best_epoch": int(_safe_float(r.get("best_epoch"))),
                "best_val_dice": round(_safe_float(r.get("best_val_dice")), 4),
                "val_best_dice_no_tta": round(_safe_float(r.get("val_best_dice_no_tta")), 4),
                "val_best_dice_tta": round(_safe_float(r.get("val_best_dice_tta")), 4),
                "threshold_tta": _safe_float(r.get("val_best_threshold_tta"), 0.5),
            }
        )

    selected_features = []
    for r in _read_csv(BRAIN_REPORTS_DIR / "final_feature_selection_table.csv"):
        selected_features.append(
            {
                "feature": r.get("feature", ""),
                "selection_count": int(_safe_float(r.get("selection_count"))),
                "combined_rank": round(_safe_float(r.get("combined_rank")), 2),
                "final_selected": r.get("final_selected") == "1",
            }
        )

    crc_phikon = crc_engine.meta_phikon
    crc_lunit = crc_engine.meta_lunit

    graphs = [
        {"file": "class_distribution.png", "title": "BRISC 2025 Class Distribution", "category": "EDA & Dataset"},
        {"file": "plane_distribution.png", "title": "MRI Anatomical Plane Coverage", "category": "EDA & Dataset"},
        {"file": "sample_mri_grid.png", "title": "Representative Brain MRI Cohort", "category": "EDA & Dataset"},
        {"file": "sample_mask_overlay_grid.png", "title": "Ground-Truth Tumor Contours", "category": "EDA & Dataset"},
        {"file": "image_intensity_by_class_boxplot.png", "title": "Intensity Distribution by Pathology", "category": "EDA & Dataset"},
        {"file": "tumor_area_by_class_boxplot.png", "title": "Tumor Burden by Pathology Class", "category": "EDA & Dataset"},
        {"file": "correlation_heatmap_triangular.png", "title": "Engineered Feature Correlations", "category": "Feature Engineering"},
        {"file": "filter_feature_scores.png", "title": "Filter Feature Selection Scores", "category": "Feature Engineering"},
        {"file": "random_forest_feature_importance.png", "title": "Random Forest Feature Importance", "category": "Classical ML"},
        {"file": "model_comparison_barplot.png", "title": "Classical ML Benchmark Comparison", "category": "Classical ML"},
        {"file": "roc_curves.png", "title": "Classical ML ROC Curves", "category": "Classical ML"},
        {"file": "confusion_matrices.png", "title": "Classical ML Confusion Matrices", "category": "Classical ML"},
        {"file": "cnn_training_curves.png", "title": "Deep CNN Optimization Curves", "category": "Deep Learning & Segmentation"},
        {"file": "cnn_confusion_matrix.png", "title": "5-Fold Ensemble Confusion Matrix", "category": "Deep Learning & Segmentation"},
        {"file": "saliency_examples.png", "title": "CNN Saliency Map Attribution", "category": "Deep Learning & Segmentation"},
        {"file": "segmentation_training_curves.png", "title": "Segmentation Dice & Loss Curves", "category": "Deep Learning & Segmentation"},
        {"file": "segmentation_examples.png", "title": "Tumor Segmentation Predictions", "category": "Deep Learning & Segmentation"},
        {"file": "risk_level_distribution.png", "title": "Clinical Risk Score Triage Distribution", "category": "Deep Learning & Segmentation"},
    ]

    return {
        "brain_tumor": {
            "dataset_summary": {
                "name": "BRISC 2025 Brain MRI Dataset",
                "total_rows": 10793,
                "classification_rows": 6000,
                "segmentation_rows": 4793,
                "train_rows": 8933,
                "test_rows": 1860,
                "classes": [
                    {"label": "Pituitary", "count": 3514},
                    {"label": "Meningioma", "count": 3270},
                    {"label": "Glioma", "count": 2802},
                    {"label": "No Tumor", "count": 1207},
                ],
                "planes": [
                    {"label": "Sagittal", "count": 3607},
                    {"label": "Coronal", "count": 3604},
                    {"label": "Axial", "count": 3582},
                ],
            },
            "cnn_ensemble_summary": {
                "strategy": "5-Fold EfficientNet-B0 Ensemble (Weighted Focal Loss)",
                "test_accuracy": 0.9930,
                "test_macro_f1": 0.9936,
                "test_macro_precision": 0.9933,
                "test_macro_recall": 0.9939,
                "per_class": cnn_per_class,
                "backbones": cnn_backbones,
            },
            "classical_ml": classical_rows,
            "segmentation": seg_models,
            "selected_features": selected_features,
            "graphs": [
                {**g, "url": f"/static/brain-graphs/{g['file']}"}
                for g in graphs
                if (BRAIN_GRAPHS_DIR / g["file"]).exists()
            ],
        },
        "colorectal": {
            "dataset_summary": {
                "name": "NCT-CRC-HE-100K & CRC-VAL-HE-7K (50 External Patients)",
                "classes": CRC_CLASS_NAMES,
                "codes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
            },
            "phikon": crc_phikon,
            "lunit_vits8": crc_lunit,
        },
    }


STUDY_BENCHMARKS = build_study_benchmarks()

CRC_SAMPLES = [
    {
        "id": "test_colorectal_adenocarcinoma_epithelium_0.png",
        "title": "Adenocarcinoma Epithelium #1",
        "true_label": "Colorectal adenocarcinoma epithelium (Tumour)",
        "category": "tumour",
        "url": "/static/crc-samples/test_colorectal_adenocarcinoma_epithelium_0.png",
    },
    {
        "id": "test_colorectal_adenocarcinoma_epithelium_1.png",
        "title": "Adenocarcinoma Epithelium #2",
        "true_label": "Colorectal adenocarcinoma epithelium (Tumour)",
        "category": "tumour",
        "url": "/static/crc-samples/test_colorectal_adenocarcinoma_epithelium_1.png",
    },
    {
        "id": "mosaic_16_patches_tumour_on_diagonal.jpg",
        "title": "16-Patch Multi-Tissue Mosaic",
        "true_label": "4 Tumour Patches on Diagonal (Tile Scan)",
        "category": "mosaic",
        "url": "/static/crc-samples/mosaic_16_patches_tumour_on_diagonal.jpg",
    },
    {
        "id": "test_normal_colon_mucosa_0.png",
        "title": "Normal Colon Mucosa",
        "true_label": "Normal colon mucosa (Benign)",
        "category": "normal",
        "url": "/static/crc-samples/test_normal_colon_mucosa_0.png",
    },
    {
        "id": "test_cancer-associated_stroma_0.png",
        "title": "Cancer-Associated Stroma",
        "true_label": "Cancer-associated stroma",
        "category": "stroma",
        "url": "/static/crc-samples/test_cancer-associated_stroma_0.png",
    },
    {
        "id": "test_lymphocytes_0.png",
        "title": "Lymphocytes Infiltrate",
        "true_label": "Lymphocytes",
        "category": "normal",
        "url": "/static/crc-samples/test_lymphocytes_0.png",
    },
    {
        "id": "test_adipose_0.png",
        "title": "Adipose Tissue",
        "true_label": "Adipose tissue",
        "category": "normal",
        "url": "/static/crc-samples/test_adipose_0.png",
    },
    {
        "id": "test_smooth_muscle_0.png",
        "title": "Smooth Muscle",
        "true_label": "Smooth muscle",
        "category": "normal",
        "url": "/static/crc-samples/test_smooth_muscle_0.png",
    },
]

BRAIN_SAMPLES = [
    {
        "id": "glioma_axial_t1.jpg",
        "title": "Glioma · Axial T1",
        "true_label": "Glioma (Axial Plane)",
        "plane": "axial",
        "category": "glioma",
        "url": "/static/brain-samples/glioma_axial_t1.jpg",
    },
    {
        "id": "meningioma_axial_t1.jpg",
        "title": "Meningioma · Axial T1",
        "true_label": "Meningioma (Axial Plane)",
        "plane": "axial",
        "category": "meningioma",
        "url": "/static/brain-samples/meningioma_axial_t1.jpg",
    },
    {
        "id": "meningioma_coronal_sample.png",
        "title": "Meningioma · Coronal T1",
        "true_label": "Meningioma (Coronal Plane)",
        "plane": "coronal",
        "category": "meningioma",
        "url": "/static/brain-samples/meningioma_coronal_sample.png",
    },
    {
        "id": "pituitary_coronal_sample.png",
        "title": "Pituitary Adenoma · Coronal",
        "true_label": "Pituitary Tumor (Coronal Plane)",
        "plane": "coronal",
        "category": "pituitary",
        "url": "/static/brain-samples/pituitary_coronal_sample.png",
    },
    {
        "id": "pituitary_axial_sample.png",
        "title": "Pituitary Adenoma · Axial",
        "true_label": "Pituitary Tumor (Axial Plane)",
        "plane": "axial",
        "category": "pituitary",
        "url": "/static/brain-samples/pituitary_axial_sample.png",
    },
    {
        "id": "no_tumor_sagittal_sample.png",
        "title": "Healthy Brain · Sagittal",
        "true_label": "No Tumor (Sagittal Plane)",
        "plane": "sagittal",
        "category": "no_tumor",
        "url": "/static/brain-samples/no_tumor_sagittal_sample.png",
    },
    {
        "id": "glioma_sagittal_sample.png",
        "title": "Glioma · Sagittal View",
        "true_label": "Glioma (Sagittal Plane)",
        "plane": "sagittal",
        "category": "glioma",
        "url": "/static/brain-samples/glioma_sagittal_sample.png",
    },
    {
        "id": "meningioma_sagittal_sample.png",
        "title": "Meningioma · Sagittal View",
        "true_label": "Meningioma (Sagittal Plane)",
        "plane": "sagittal",
        "category": "meningioma",
        "url": "/static/brain-samples/meningioma_sagittal_sample.png",
    },
]

NON_MEDICAL_KEYWORDS = {
    "poem",
    "recipe",
    "cooking",
    "football",
    "soccer",
    "basketball",
    "stock market",
    "crypto",
    "bitcoin",
    "movie",
    "song",
    "lyrics",
    "weather",
    "horoscope",
    "joke",
    "video game",
}

REFUSAL_MESSAGE = (
    "I can only help with medical and health-related questions, such as pathology, colorectal cancer, "
    "neuro-oncology, brain MRI analysis, medical imaging, or your MedVisionAI diagnostic reports. "
    "Feel free to ask me something in that area."
)


def generate_clinical_assistant_response(
    message: str,
    attached_case: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    q_lower = message.lower().strip()
    if any(k in q_lower for k in NON_MEDICAL_KEYWORDS) and not any(
        m in q_lower for m in ("tumor", "tumour", "cancer", "mri", "tissue", "colon", "brain", "pathology", "patient", "health", "medical")
    ):
        return {
            "reply": REFUSAL_MESSAGE,
            "provider": "MedVisionAI Clinical Scope Guard",
        }

    if attached_case:
        modality = attached_case.get("modality", "")
        fname = attached_case.get("filename", "the attached scan")
        if modality == "brain_mri":
            cls_info = attached_case.get("classification", {})
            seg_info = attached_case.get("segmentation", {})
            risk_info = attached_case.get("risk_summary", {})
            classical_info = attached_case.get("classical_support", {})
            pred_cls = cls_info.get("predicted_class", "unknown").replace("_", " ").title()
            prob = cls_info.get("probability", 0.0) * 100
            area_pct = seg_info.get("tumor_area_percent", 0.0)
            risk_score = risk_info.get("risk_score", 0.0)
            risk_level = risk_info.get("risk_level", "low").upper()
            priority = risk_info.get("suggested_priority", "routine review")

            backbones_lines = []
            for bkey, bval in (cls_info.get("all_backbones") or {}).items():
                backbones_lines.append(
                    f"- **{bval.get('title', bkey)}**: `{bval.get('predicted_class')}` ({bval.get('probability', 0)*100:.1f}%)"
                )
            backbones_md = "\n".join(backbones_lines)

            classical_lines = []
            for cval in (classical_info.get("all_classical_models") or [])[:4]:
                classical_lines.append(
                    f"- **{cval['model_name']}**: `{cval['predicted_binary_label']}` ({cval['tumor_probability']*100:.1f}% tumor probability)"
                )
            classical_md = "\n".join(classical_lines)

            reply = (
                f"### Clinical Case Synthesis: `{fname}`\n\n"
                f"**1. Primary Neuro-Oncology Finding**\n"
                f"- **Predicted Class:** **{pred_cls}** with **{prob:.2f}%** confidence using `{cls_info.get('selected_model_title', '5-Fold EfficientNet-B0 Ensemble')}`.\n"
                f"- **Anatomical Plane:** `{attached_case.get('plane', 'axial').title()}` view.\n\n"
                f"**2. Multi-Backbone Deep Learning Consensus**\n"
                f"{backbones_md}\n\n"
                f"**3. Volumetric Segmentation & Morphology (`UNetSmall`)**\n"
                f"- **Estimated Tumor Burden:** **{area_pct:.2f}%** of the slice field of view ({seg_info.get('tumor_area_px', 0):.0f} px, size category: `{risk_info.get('tumor_size_category', 'unknown')}`).\n"
                f"- **Intensity Contrast:** `{risk_info.get('intensity_contrast', 0.0):.2f}` (inside vs. outside lesion mean intensity).\n"
                f"- **Saliency & Segmentation Overlays:** The indigo contour highlights the predicted lesion boundary, while the magma heatmap shows spatial probability density.\n\n"
                f"**4. Classical ML Feature Support (7 Engineered Features)**\n"
                f"{classical_md}\n\n"
                f"**5. Triage Priority & Next Steps**\n"
                f"- **Clinical Risk Score:** **{risk_score}/100 ({risk_level})** — *{priority}*.\n"
                f"- **Recommended Action:** Correlate with multi-sequence MRI (T1+C, T2, FLAIR) and refer to Neuro-Oncology or Neuroradiology via the **Appointments & Triage** tab.\n\n"
                f"> *Disclaimer: MedVisionAI is a research decision-support prototype and not a clinical diagnostic device.*"
            )
            return {
                "reply": reply,
                "provider": "MedVisionAI Neuro-Oncology & Radiology Knowledge Engine",
            }

        elif modality == "colorectal":
            finding = attached_case.get("finding", "Unknown")
            pred_cls = attached_case.get("predicted_class") or "Not assessed"
            pred_prob = (attached_case.get("predicted_probability") or 0.0) * 100
            tum_prob = (attached_case.get("tumor_probability") or 0.0) * 100
            conf = attached_case.get("confidence", "Unknown")
            mode_str = attached_case.get("analysis_mode", "single patch")
            reasons = "\n".join(f"- {r}" for r in (attached_case.get("confidence_reasons") or []))

            top_classes = sorted(
                (attached_case.get("class_probabilities") or {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:4]
            top_md = "\n".join(f"- **{k}**: {v*100:.2f}%" for k, v in top_classes)

            tiles_md = ""
            if attached_case.get("tiles"):
                t = attached_case["tiles"]
                tiles_md = (
                    f"\n\n**Tile-Scan Breakdown**\n"
                    f"- Scanned **{t['n_tiles']} tiles** ({t['tile_size_px']} px each); **{t['n_tissue_tiles']}** contained stained tissue.\n"
                    f"- Flagged **{t['n_tumor_tiles']} tumour tiles** ({t['tumor_fraction']*100:.1f}% of tissue tiles; threshold ≥ {t['required_tumor_fraction']*100:.0f}%)."
                )

            reply = (
                f"### Histopathology Case Synthesis: `{fname}`\n\n"
                f"**1. Primary Finding**\n"
                f"- **Verdict:** **{finding}** (Confidence: **{conf}**)\n"
                f"- **Dominant Tissue Type:** **{pred_cls}** ({pred_prob:.2f}% calibrated probability)\n"
                f"- **Adenocarcinoma (Tumour) Probability:** **{tum_prob:.2f}%**\n"
                f"- **Analysis Mode:** `{mode_str}`\n\n"
                f"**2. Calibrated Tissue Probability Breakdown (Top Classes)**\n"
                f"{top_md}"
                f"{tiles_md}\n\n"
                f"**3. Reading the Token Grad-CAM Heatmap**\n"
                f"- The Grad-CAM overlay measures gradient attention across the Vision Transformer patch tokens entering the final transformer block for **{attached_case.get('heatmap_target') or pred_cls}**.\n"
                f"- Red-to-yellow regions indicate cellular structures that strongly drove the classification; uncolored regions had minimal influence.\n\n"
                f"**4. Uncertainty & Quality Assessment**\n"
                f"{reasons}\n\n"
                f"> *Disclaimer: MedVisionAI is a research prototype trained on NCT-CRC-HE-100K. Always confirm findings with a board-certified gastrointestinal pathologist.*"
            )
            return {
                "reply": reply,
                "provider": "MedVisionAI Computational Pathology Engine",
            }

    if "glioma" in q_lower or "meningioma" in q_lower or "pituitary" in q_lower or "brain" in q_lower or "mri" in q_lower:
        return {
            "reply": (
                "### Brain Tumor MRI Classification & Segmentation in MedVisionAI\n\n"
                "Our neuro-oncology pipeline analyzes T1-weighted brain MRI slices across **axial, coronal, and sagittal** planes:\n\n"
                "- **Glioma:** Arises from glial cells within the brain parenchyma. On MRI, gliomas often show irregular margins, heterogeneous intensity, and surrounding perilesional edema. Assigned the highest baseline clinical triage weight (`25 pts`).\n"
                "- **Meningioma:** Extra-axial neoplasms arising from arachnoid cap cells of the meninges. Typically well-circumscribed, homogeneously enhancing dural-based masses (`12 pts` baseline risk weight).\n"
                "- **Pituitary Adenoma:** Sellar and suprasellar masses arising from the anterior pituitary gland, frequently evaluated on coronal and sagittal T1 sequences (`18 pts` baseline risk weight).\n"
                "- **Hybrid Architecture:** MedVisionAI combines a **5-fold EfficientNet-B0 ensemble** (99.3% test accuracy on BRISC 2025), single backbones (**ConvNeXt-Tiny**, **ResNet-18**), a **UNetSmall segmentation network** for lesion contouring, and **5 classical ML models** trained on 7 engineered radiomic/intensity features (`blur_score`, `edge_density`, `image_std_intensity`, `image_mean_intensity`, `image_entropy`, `plane_code`, `image_quality_score`)."
            ),
            "provider": "MedVisionAI Neuro-Oncology Knowledge Base",
        }

    if "stroma" in q_lower or "adenocarcinoma" in q_lower or "mucosa" in q_lower or "colorectal" in q_lower or "h&e" in q_lower:
        return {
            "reply": (
                "### Colorectal Histopathology (NCT-CRC-HE-100K 9-Class System)\n\n"
                "MedVisionAI classifies H&E-stained colorectal tissue patches into 9 distinct morphological classes:\n\n"
                "- **Colorectal Adenocarcinoma Epithelium (`TUM`):** Malignant glandular proliferation characterized by nuclear stratification, loss of polarity, hyperchromasia, and irregular cribriform or infiltrative glands.\n"
                "- **Normal Colon Mucosa (`NORM`):** Regular, evenly spaced crypts lined by uniform goblet cells and absorptive colonocytes with basal nuclei.\n"
                "- **Cancer-Associated Stroma (`STR`):** Desmoplastic fibrous tissue surrounding invasive tumor glands, rich in activated myofibroblasts and extracellular matrix.\n"
                "- **Other Non-Tumour Classes:** Lymphocytes (`LYM`), Debris/Necrosis (`DEB`), Mucus (`MUC`), Smooth Muscle (`MUS`), Adipose Tissue (`ADI`), and Background (`BACK`).\n\n"
                "Probabilities are temperature-calibrated on external validation patches (**CRC-VAL-HE-7K**, 50 patients), and spatial attribution is visualized via **ViT Token Grad-CAM**."
            ),
            "provider": "MedVisionAI Histopathology Knowledge Base",
        }

    if "grad-cam" in q_lower or "saliency" in q_lower or "heatmap" in q_lower or "explain" in q_lower:
        return {
            "reply": (
                "### Interpreting Grad-CAM & Saliency Maps in MedVisionAI\n\n"
                "- **ViT Token Grad-CAM (Colorectal H&E):** Computes gradients of the target tissue logit with respect to patch tokens entering the final Vision Transformer LayerNorm. Warm yellow/red zones highlight the exact 8×8 or 16×16 tissue regions that drove the linear head's prediction.\n"
                "- **Input-Gradient Saliency & U-Net Probability Maps (Brain MRI):** For brain MRI scans, MedVisionAI produces both pixel-level input-gradient saliency from the CNN classifier and a continuous spatial probability map (`UNetSmall`) to estimate tumor area percentage, bounding box compactness, and inside/outside intensity contrast.\n"
                "- **Clinical Caveat:** Heatmaps indicate model attention rather than definitive histological or radiological boundaries."
            ),
            "provider": "MedVisionAI Explainable AI Guide",
        }

    return {
        "reply": (
            "### MedVisionAI Clinical Intelligence Assistant\n\n"
            "I can assist you with:\n"
            "- **Brain Tumor MRI Diagnostics:** Interpreting Glioma, Meningioma, Pituitary Adenoma, and Healthy MRI scans, comparing our 5-fold EfficientNet-B0 ensemble against ConvNeXt-Tiny, ResNet-18, and Classical ML models, and reviewing U-Net lesion segmentation and clinical risk scores.\n"
            "- **Colorectal Histopathology:** Interpreting 9-class H&E tissue predictions, temperature-calibrated probabilities, multi-patch tile scans, and ViT Token Grad-CAM overlays.\n"
            "- **Triage & Specialist Referrals:** Determining urgency levels (*Urgent*, *Soon*, *Routine*) for Neuro-Oncology, Neuroradiology, or Gastrointestinal Pathology consultations.\n\n"
            "Select an analyzed case from the dropdown above or ask a specific question about pathology, radiology, or model performance!"
        ),
        "provider": "MedVisionAI Clinical Intelligence Engine",
    }


@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon() -> FileResponse:
    fav_svg = FRONTEND_DIR / "favicon.svg"
    if fav_svg.exists():
        return FileResponse(fav_svg, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "device": str(DEVICE),
            "engines": {
                "colorectal_histopathology": {
                    "loaded": True,
                    "classes": CRC_CLASS_NAMES,
                    "backbone": "Lunit DINO ViT-S/8 + Phikon ViT-B/16",
                },
                "brain_tumor_mri": {
                    "loaded": True,
                    "classes": brain_engine.class_names,
                    "cnn_strategy": brain_engine.cnn_strategy_name,
                    "cnn_backbones": ["ensemble"] + list(brain_engine.single_backbones.keys()),
                    "segmentation_model": brain_engine.segmentation_model_name,
                    "classical_models": list(brain_engine.classical_models.keys()),
                },
            },
        }
    )


@app.get("/api/samples")
async def list_samples() -> JSONResponse:
    return JSONResponse(
        {
            "colorectal": CRC_SAMPLES,
            "brain_mri": BRAIN_SAMPLES,
        }
    )


@app.get("/api/benchmarks")
async def get_benchmarks() -> JSONResponse:
    return JSONResponse(STUDY_BENCHMARKS)


@app.post("/api/analyze/colorectal")
async def analyze_colorectal_endpoint(
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    mode: str = Form("Auto"),
    tile_size: str = Form("Auto"),
    min_tumor_fraction: float = Form(0.10),
    heatmap_alpha: float = Form(0.50),
    force_out_of_scope: bool = Form(False),
) -> JSONResponse:
    if file is not None:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.")
        filename = file.filename or "uploaded_histology.png"
        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc
    elif sample_id:
        sample_path = CRC_EXAMPLES_DIR / Path(sample_id).name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        pil_image = Image.open(sample_path).convert("RGB")
        filename = sample_path.name
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded file or a sample_id.")

    parsed_tile = None if tile_size in ("Auto", "", "null", "None") else int(tile_size)
    result = crc_engine.analyze(
        pil_image=pil_image,
        filename=filename,
        mode=mode,
        tile_size=parsed_tile,
        min_tumor_fraction=float(min_tumor_fraction),
        heatmap_alpha=float(heatmap_alpha),
        force_out_of_scope=bool(force_out_of_scope),
    )
    return JSONResponse(result)


@app.post("/api/analyze/brain")
async def analyze_brain_endpoint(
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    plane: str = Form("auto"),
    backbone: str = Form("ensemble"),
    segmentation_threshold: float = Form(0.35),
    use_tta: bool = Form(True),
) -> JSONResponse:
    if file is not None:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.")
        filename = file.filename or "uploaded_mri.jpg"
        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid MRI image file: {exc}") from exc
    elif sample_id:
        sample_path = BRAIN_SAMPLES_DIR / Path(sample_id).name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Brain MRI sample '{sample_id}' not found.")
        pil_image = Image.open(sample_path).convert("RGB")
        filename = sample_path.name
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded MRI file or a sample_id.")

    result = brain_engine.analyze(
        pil_image=pil_image,
        filename=filename,
        selected_plane=plane,
        backbone_choice=backbone,
        segmentation_threshold=float(segmentation_threshold),
        use_tta=bool(use_tta),
    )
    return JSONResponse(result)


class ChatRequest(BaseModel):
    message: str
    attached_case: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = None


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest) -> JSONResponse:
    resp = generate_clinical_assistant_response(
        message=payload.message,
        attached_case=payload.attached_case,
        history=payload.history,
    )
    return JSONResponse(resp)


@app.get("/api/appointments")
async def list_appointments() -> JSONResponse:
    return JSONResponse(
        {
            "specialists": SPECIALISTS,
            "urgency_levels": URGENCY_INFO,
            "appointments": _with_queue_positions(APPOINTMENTS_STORE),
        }
    )


class CreateAppointmentRequest(BaseModel):
    specialist_id: str
    urgency: str
    reason: str
    requester_name: str
    contact: str
    organisation: str = "August Clinical Partner"
    preferred: str = ""
    patient_ref: str = ""
    case_summary: str = ""


@app.post("/api/appointments")
async def create_appointment(payload: CreateAppointmentRequest) -> JSONResponse:
    spec = next((s for s in SPECIALISTS if s["id"] == payload.specialist_id), SPECIALISTS[0])
    urg_key = payload.urgency if payload.urgency in URGENCY_INFO else "routine"
    if not payload.requester_name.strip() or not payload.contact.strip() or not payload.reason.strip():
        raise HTTPException(status_code=400, detail="Requester name, contact, and clinical reason are required.")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_row = {
        "id": f"req-{uuid.uuid4().hex[:8]}",
        "created_at": now,
        "updated_at": now,
        "organisation": payload.organisation.strip() or "August Clinical Partner",
        "requester_name": payload.requester_name.strip(),
        "contact": payload.contact.strip(),
        "hospital": spec["hospital"],
        "department": spec["department"],
        "specialist": spec["title"],
        "specialist_id": spec["id"],
        "urgency": urg_key,
        "priority": URGENCY_INFO[urg_key]["priority"],
        "reason": payload.reason.strip(),
        "preferred": payload.preferred.strip(),
        "patient_ref": payload.patient_ref.strip(),
        "case_summary": payload.case_summary.strip(),
        "status": "pending",
        "appointment_at": None,
        "location": "",
        "staff_note": "",
    }
    APPOINTMENTS_STORE.append(new_row)
    all_with_pos = _with_queue_positions(APPOINTMENTS_STORE)
    created = next((r for r in all_with_pos if r["id"] == new_row["id"]), new_row)
    return JSONResponse({"appointment": created, "appointments": all_with_pos})


class UpdateAppointmentRequest(BaseModel):
    status: str
    appointment_at: str | None = None
    location: str | None = None
    staff_note: str | None = None


@app.patch("/api/appointments/{appointment_id}")
async def update_appointment(appointment_id: str, payload: UpdateAppointmentRequest) -> JSONResponse:
    target = next((r for r in APPOINTMENTS_STORE if r["id"] == appointment_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Appointment request not found.")
    valid_statuses = {"pending", "confirmed", "completed", "cancelled"}
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status '{payload.status}'. Must be one of {valid_statuses}.")
    target["status"] = payload.status
    target["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if payload.appointment_at is not None:
        target["appointment_at"] = payload.appointment_at
    if payload.location is not None:
        target["location"] = payload.location
    if payload.staff_note is not None:
        target["staff_note"] = payload.staff_note
    all_with_pos = _with_queue_positions(APPOINTMENTS_STORE)
    return JSONResponse({"appointment": target, "appointments": all_with_pos})


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
