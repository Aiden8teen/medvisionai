"""Unified ML Engine for MedVisionAI: Colorectal Histopathology + Brain Tumor MRI."""
from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import struct
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

ROOT_DIR = Path(os.environ.get("MEDVISION_ROOT", Path(__file__).resolve().parent.parent))
MEDVISION_REPO_DIR = ROOT_DIR / "medvisionai"
BRAINTUMOR_DIR = ROOT_DIR / "braintumor"

CRC_MODELS_DIR = MEDVISION_REPO_DIR / "models"
CRC_EXAMPLES_DIR = MEDVISION_REPO_DIR / "examples"

BRAIN_MODELS_DIR = BRAINTUMOR_DIR / "models"
BRAIN_SAMPLES_DIR = BRAINTUMOR_DIR / "sample_mris"
BRAIN_REPORTS_DIR = BRAINTUMOR_DIR / "submissionwithnomodels" / "reports"
BRAIN_DATA_DIR = BRAINTUMOR_DIR / "submissionwithnomodels" / "data_cleaned"
BRAIN_GRAPHS_DIR = BRAINTUMOR_DIR / "submissionwithnomodels" / "graphs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        f.seek(0)
        buf = f.read()
    sd: dict[str, torch.Tensor] = {}
    dtype_map = {
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
    }
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dt = dtype_map[meta["dtype"]]
        count = (end - start) // np.dtype(dt).itemsize
        arr = np.frombuffer(buf, dtype=dt, count=count, offset=data_start + start)
        sd[k] = torch.from_numpy(arr.copy().reshape(meta["shape"]))
    return sd


def image_to_base64(image_array: np.ndarray | Image.Image, mode: str = "rgb") -> str:
    if isinstance(image_array, Image.Image):
        image = image_array
    elif mode == "gray":
        image = Image.fromarray(image_array.astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


CRC_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CRC_CLASS_NAMES = [
    "Adipose tissue",
    "Background",
    "Debris / necrosis",
    "Lymphocytes",
    "Mucus",
    "Smooth muscle",
    "Normal colon mucosa",
    "Cancer-associated stroma",
    "Colorectal adenocarcinoma epithelium",
]
CRC_TUMOR_INDEX = CRC_CODES.index("TUM")


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int = 384, hidden_features: int = 1536):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerTimm(nn.Module):
    def __init__(self, patch_size: int = 8, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.num_features = embed_dim
        self.num_prefix_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.patch_embed = PatchEmbed(224, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class LinearHead(nn.Module):
    def __init__(self, dim: int = 384, n_classes: int = 9):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("temperature", torch.ones(()))
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc((feats - self.mean) / self.std) / self.temperature


class TokenGradCAM:
    def __init__(self, model: VisionTransformerTimm):
        self.model = model
        self.act: torch.Tensor | None = None
        self.grad: torch.Tensor | None = None
        self._h = model.blocks[-1].norm1.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.act = output
        output.register_hook(lambda g: setattr(self, "grad", g))

    def remove(self) -> None:
        self._h.remove()

    def cam(self) -> np.ndarray:
        n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        assert self.act is not None and self.grad is not None
        act, grad = self.act[:, n_prefix:], self.grad[:, n_prefix:]
        side = int(math.isqrt(act.shape[1]))
        w = grad.mean(dim=1, keepdim=True)
        cam = F.relu((w * act).sum(-1)).reshape(-1, side, side)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)


def _tissue_mask(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    return (sat > 0.08) & (mx < 0.94) & (mx > 0.08)


def assess_histology_input(img: Image.Image) -> dict[str, Any]:
    small = img.copy()
    small.thumbnail((384, 384), Image.LANCZOS)
    arr = np.asarray(small.convert("RGB"), dtype=np.uint8)
    rgb = arr.astype(np.float32) / 255.0
    chroma = rgb.max(-1) - rgb.min(-1)
    colourfulness = float(chroma.mean())
    tissue = _tissue_mask(arr)
    tissue_fraction = float(tissue.mean())

    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    hue_deg = hsv[..., 0] * 360.0 / 255.0
    he_hue = (hue_deg >= 250) | (hue_deg <= 25)
    he_fraction = float(he_hue[tissue].mean()) if tissue.any() else 0.0

    warnings: list[str] = []
    if colourfulness < 0.02 or (colourfulness < 0.05 and tissue_fraction < 0.03):
        image_type, applicable = "grayscale (e.g. radiology / MRI)", False
        warnings.append(
            "The image is nearly grayscale (like an MRI, CT, or X-ray). The Colorectal Histopathology model "
            "is trained on colour H&E tissue patches; try switching to the Brain Tumor MRI modality."
        )
    elif tissue_fraction < 0.05:
        image_type, applicable = "little stained tissue", True
        warnings.append("Very little stained tissue was found; the image is mostly background, fat, or mucus.")
    elif he_fraction < 0.35:
        image_type, applicable = "unusual colours", False
        warnings.append(
            "The colours do not look like an H&E stain (pink/purple). The image may be a different stain or modality."
        )
    else:
        image_type, applicable = "H&E-like histology", True
        if he_fraction < 0.6:
            warnings.append("The staining colours are only partly H&E-like; the result may be less reliable.")
    if min(img.size) < 64:
        warnings.append("The image is very small, so spatial detail is limited.")
    return {
        "image_type": image_type,
        "applicable": applicable,
        "tissue_fraction": round(tissue_fraction, 3),
        "he_colour_fraction": round(he_fraction, 3),
        "colourfulness": round(colourfulness, 3),
        "warnings": warnings,
    }


def tile_boxes(width: int, height: int, tile: int) -> list[tuple[int, int, int, int]]:
    tile = max(1, min(tile, width, height))
    nx, ny = math.ceil(width / tile), math.ceil(height / tile)
    boxes = []
    for j in range(ny):
        y0 = min(j * tile, height - tile)
        for i in range(nx):
            x0 = min(i * tile, width - tile)
            boxes.append((x0, y0, x0 + tile, y0 + tile))
    return boxes


def overlay_gradcam(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    c_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    c = np.asarray(c_img, dtype=np.float32) / 255.0
    heat = np.stack([np.ones_like(c), np.clip(2 * c - 0.6, 0, 1), np.zeros_like(c)], -1)
    a = (alpha * np.clip((c - 0.15) / 0.85, 0, 1))[..., None]
    return Image.fromarray((((1 - a) * base + a * heat).clip(0, 1) * 255).astype(np.uint8))


class ColorectalEngine:
    def __init__(self) -> None:
        self.backbone = VisionTransformerTimm(patch_size=8, embed_dim=384, depth=12, num_heads=6)
        safetensors_path = CRC_MODELS_DIR / "lunit_vits8.safetensors"
        self.backbone.load_state_dict(load_safetensors(safetensors_path), strict=True)
        self.backbone.to(DEVICE).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        ck_lunit = torch.load(CRC_MODELS_DIR / "crc_head_lunit_vits8.pt", map_location=DEVICE, weights_only=False)
        self.head_lunit = LinearHead(384, 9)
        self.head_lunit.load_state_dict(ck_lunit["head"], strict=True)
        self.head_lunit.to(DEVICE).eval()
        self.meta_lunit = ck_lunit.get("meta", {})

        ck_phikon = torch.load(CRC_MODELS_DIR / "crc_head_phikon.pt", map_location=DEVICE, weights_only=False)
        self.meta_phikon = ck_phikon.get("meta", {})

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def explain_patches(
        self, images: list[Image.Image], class_index: int | None = None, batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray]:
        probs_list, cams_list = [], []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            x = torch.stack([self.transform(im.convert("RGB")) for im in batch_imgs]).to(DEVICE)
            engine = TokenGradCAM(self.backbone)
            try:
                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    logits = self.head_lunit(self.backbone(x))
                    target = (
                        logits.argmax(1)
                        if class_index is None
                        else torch.full((len(x),), class_index, dtype=torch.long, device=DEVICE)
                    )
                    logits.gather(1, target[:, None]).sum().backward()
                cams_list.append(engine.cam())
            finally:
                engine.remove()
            probs_list.append(torch.softmax(logits.detach(), 1).cpu().numpy())
        return np.concatenate(probs_list), np.concatenate(cams_list)

    def analyze(
        self,
        pil_image: Image.Image,
        filename: str = "uploaded_patch.png",
        mode: str = "Auto",
        tile_size: int | None = None,
        min_tumor_fraction: float = 0.10,
        heatmap_alpha: float = 0.50,
        force_out_of_scope: bool = False,
    ) -> dict[str, Any]:
        w, h = pil_image.size
        check = assess_histology_input(pil_image)
        names, T = CRC_CLASS_NAMES, CRC_TUMOR_INDEX

        display_img = pil_image.copy()
        display_img.thumbnail((896, 896), Image.LANCZOS)

        if not check["applicable"] and not force_out_of_scope:
            return {
                "modality": "colorectal",
                "filename": filename,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "image_size": [w, h],
                "finding": "Not assessed",
                "applicable": False,
                "analysis_mode": "none",
                "predicted_class": None,
                "predicted_probability": None,
                "tumor_probability": None,
                "class_probabilities": {},
                "confidence": "Not assessed",
                "confidence_reasons": check["warnings"]
                + ["No tissue classification was made because the image is outside the H&E colorectal domain."],
                "heatmap_target": None,
                "input_check": check,
                "tiles": None,
                "preview_images": {
                    "original": image_to_base64(display_img),
                    "overlay": None,
                    "heatmap_only": None,
                },
                "model_info": {
                    "name": "MedVisionAI Colorectal Histopathology Classifier",
                    "backbone": "Lunit DINO ViT-S/8 + Calibrated Linear Head (with Phikon ViT-B/16 benchmark)",
                    "classes": len(names),
                    "test_metrics": self.meta_lunit.get("test", {}),
                    "phikon_metrics": self.meta_phikon.get("test", {}),
                },
            }

        extra = list(check["warnings"])
        if not check["applicable"]:
            extra.append("Analysis was forced on an image outside the H&E histology domain; interpret with caution.")

        actual_mode = mode
        if actual_mode == "Auto":
            actual_mode = "Single patch" if max(w, h) <= 336 else "Tile scan"

        if actual_mode == "Single patch":
            probs, cams = self.explain_patches([pil_image])
            p = probs[0]
            top = int(p.argmax())
            cam = cams[0]
            top_prob = float(p[top])
            level = 2 if top_prob >= 0.9 else 1 if top_prob >= 0.7 else 0
            reasons = [
                {
                    2: f"The model is decisive ({top_prob:.1%} calibrated probability for {names[top]}).",
                    1: f"The model is moderately confident ({top_prob:.1%} for {names[top]}).",
                    0: f"The model has low certainty ({top_prob:.1%} for {names[top]}); multiple tissue types are plausible.",
                }[level]
            ]
            if extra:
                level = max(0, level - 1)
                reasons.extend(extra)
            conf_label = ["Low", "Moderate", "High"][level]
            finding = "Tumour tissue detected" if top == T else "No tumour tissue detected"
            overlay_img = overlay_gradcam(display_img, cam, alpha=heatmap_alpha)
            heat_uint8 = np.uint8(np.clip(cv2.resize(cam, display_img.size), 0, 1) * 255)
            heat_color = cv2.cvtColor(cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

            return {
                "modality": "colorectal",
                "filename": filename,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "image_size": [w, h],
                "finding": finding,
                "applicable": check["applicable"],
                "analysis_mode": "Single patch (224×224 px)",
                "predicted_class": names[top],
                "predicted_probability": round(top_prob, 4),
                "tumor_probability": round(float(p[T]), 4),
                "class_probabilities": {n: round(float(v), 4) for n, v in zip(names, p)},
                "confidence": conf_label,
                "confidence_reasons": reasons,
                "heatmap_target": names[top],
                "input_check": check,
                "tiles": None,
                "preview_images": {
                    "original": image_to_base64(display_img),
                    "overlay": image_to_base64(overlay_img),
                    "heatmap_only": image_to_base64(heat_color),
                },
                "model_info": {
                    "name": "MedVisionAI Colorectal Histopathology Classifier",
                    "backbone": "Lunit DINO ViT-S/8 + Calibrated Linear Head",
                    "classes": len(names),
                    "test_metrics": self.meta_lunit.get("test", {}),
                    "phikon_metrics": self.meta_phikon.get("test", {}),
                },
            }

        tile = tile_size or max(224, math.ceil(max(w, h) / 4))
        tile = min(tile, w, h)
        boxes = tile_boxes(w, h, tile)
        probs, cams = self.explain_patches([pil_image.crop(b) for b in boxes], class_index=T)
        top = probs.argmax(1)
        background_idx = names.index("Background")
        tissue = top != background_idx
        n_tissue = int(tissue.sum())
        tumor_tiles = top == T
        n_tumor = int(tumor_tiles.sum())
        needed = max(1, math.ceil(min_tumor_fraction * max(n_tissue, 1)))
        is_tumor = n_tumor >= needed

        scale = min(1.0, 768 / max(w, h))
        cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
        cam_sum = np.zeros((ch, cw), np.float32)
        cam_cnt = np.zeros((ch, cw), np.float32)
        for (x0, y0, x1, y1), c, pt in zip(boxes, cams, probs[:, T]):
            x0, y0, x1, y1 = (round(v * scale) for v in (x0, y0, x1, y1))
            sw, sh = max(1, x1 - x0), max(1, y1 - y0)
            r = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((sw, sh), Image.BICUBIC), np.float32) / 255.0
            cam_sum[y0 : y0 + sh, x0 : x0 + sw] += (r * pt)[: ch - y0, : cw - x0]
            cam_cnt[y0 : y0 + sh, x0 : x0 + sw] += 1
        cam = cam_sum / np.maximum(cam_cnt, 1)

        composition = {n: round(float((top == k).mean()), 4) for k, n in enumerate(names)}
        mean_probs = probs.mean(0)
        decisive = float(probs[tumor_tiles, T].mean()) if is_tumor else float(1 - probs[:, T].max())
        extra.append("Tile scan over a multi-patch field of view; evaluated across grid tiles.")
        level = 2 if decisive >= 0.9 else 1 if decisive >= 0.7 else 0
        reasons = [f"Tile-scan decisive score: {decisive:.1%}."] + extra
        conf_label = ["Low", "Moderate", "High"][max(0, level - 1 if extra else level)]
        k_top = int(mean_probs.argmax())
        overlay_img = overlay_gradcam(display_img, cam, alpha=heatmap_alpha)
        heat_uint8 = np.uint8(np.clip(cv2.resize(cam, display_img.size), 0, 1) * 255)
        heat_color = cv2.cvtColor(cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

        return {
            "modality": "colorectal",
            "filename": filename,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "image_size": [w, h],
            "finding": "Tumour tissue detected" if is_tumor else "No tumour tissue detected",
            "applicable": check["applicable"],
            "analysis_mode": f"Tile scan ({len(boxes)} tiles of {tile} px)",
            "predicted_class": names[k_top],
            "predicted_probability": round(float(mean_probs[k_top]), 4),
            "tumor_probability": round(float(probs[:, T].max()), 4),
            "class_probabilities": {n: round(float(v), 4) for n, v in zip(names, mean_probs)},
            "confidence": conf_label,
            "confidence_reasons": reasons,
            "heatmap_target": names[T],
            "input_check": check,
            "tiles": {
                "tile_size_px": tile,
                "n_tiles": len(boxes),
                "n_tissue_tiles": n_tissue,
                "n_tumor_tiles": n_tumor,
                "tumor_fraction": round(n_tumor / max(n_tissue, 1), 4),
                "required_tumor_fraction": min_tumor_fraction,
                "composition": composition,
                "rule": f"Flagged when at least {needed} of {n_tissue} tissue tiles are classified as tumour (≥ {min_tumor_fraction:.0%}).",
            },
            "preview_images": {
                "original": image_to_base64(display_img),
                "overlay": image_to_base64(overlay_img),
                "heatmap_only": image_to_base64(heat_color),
            },
            "model_info": {
                "name": "MedVisionAI Colorectal Histopathology Classifier",
                "backbone": "Lunit DINO ViT-S/8 + Calibrated Linear Head",
                "classes": len(names),
                "test_metrics": self.meta_lunit.get("test", {}),
                "phikon_metrics": self.meta_phikon.get("test", {}),
            },
        }


BRAIN_CLASS_LABELS = ["glioma", "meningioma", "pituitary", "no_tumor"]
PLANE_CODES = {"unknown": 0, "axial": 1, "coronal": 2, "sagittal": 3}
CLASS_RISK_WEIGHTS = {"no_tumor": 0, "meningioma": 12, "pituitary": 18, "glioma": 25}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bridge = self.bridge(self.pool2(d2))
        u2 = self.up2(bridge)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        return self.out(u1)


def build_classifier_model(backbone_name: str, num_classes: int) -> nn.Module:
    if backbone_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if backbone_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier backbone: {backbone_name}")


def classifier_forward_probs(model: nn.Module, images: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
    logits = model(images)
    probs = torch.softmax(logits, dim=1)
    if not use_tta:
        return probs
    flipped_images = torch.flip(images, dims=[3])
    flipped_probs = torch.softmax(model(flipped_images), dim=1)
    return (probs + flipped_probs) / 2.0


def load_numeric_ranges() -> dict[str, tuple[float, float]]:
    csv_path = BRAIN_DATA_DIR / "brisc_features_cleaned.csv"
    targets = [
        "blur_score",
        "image_entropy",
        "image_std_intensity",
        "tumor_area_percent",
        "intensity_contrast",
    ]
    values: dict[str, list[float]] = {key: [] for key in targets}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in targets:
                    raw = row.get(key, "")
                    if raw in {"", "nan", "NaN", None}:
                        continue
                    try:
                        values[key].append(float(raw))
                    except ValueError:
                        continue
    ranges = {}
    for key, samples in values.items():
        ranges[key] = (float(min(samples)), float(max(samples))) if samples else (0.0, 1.0)
    return ranges


def normalize_value(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return float(np.clip((value - min_value) / (max_value - min_value), 0.0, 1.0))


def shannon_entropy(gray_image: np.ndarray) -> float:
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def blur_score(gray_image: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def edge_density(gray_image: np.ndarray) -> float:
    edges = cv2.Canny(gray_image, 100, 200)
    return float(edges.mean() / 255.0)


def parse_plane(filename: str, selected_plane: str) -> str:
    if selected_plane and selected_plane != "auto":
        return selected_plane
    stem = Path(filename).stem.lower()
    tokens = [token for token in stem.replace("-", "_").split("_") if token]
    if any(token in {"axial", "ax"} for token in tokens):
        return "axial"
    if any(token in {"coronal", "cor", "co"} for token in tokens):
        return "coronal"
    if any(token in {"sagittal", "sag", "sa"} for token in tokens):
        return "sagittal"
    return "axial"


def compute_mask_features(gray_image: np.ndarray, mask_binary: np.ndarray) -> dict[str, Any]:
    mask_binary = (mask_binary > 0).astype(np.uint8)
    area = int(mask_binary.sum())
    total = int(mask_binary.size)
    result: dict[str, Any] = {
        "tumor_area_px": float(area),
        "tumor_area_ratio": float(area / max(total, 1)),
        "bbox_x": None,
        "bbox_y": None,
        "bbox_width": None,
        "bbox_height": None,
        "bbox_area": 0.0,
        "centroid_x": None,
        "centroid_y": None,
        "compactness": None,
        "mask_mean_intensity_inside_tumor": None,
        "mask_mean_intensity_outside_tumor": None,
    }
    if area == 0:
        if np.any(mask_binary == 0):
            result["mask_mean_intensity_outside_tumor"] = float(gray_image[mask_binary == 0].mean())
        return result

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea) if contours else None
    if biggest is not None:
        x, y, w, h = cv2.boundingRect(biggest)
        result.update(
            {
                "bbox_x": float(x),
                "bbox_y": float(y),
                "bbox_width": float(w),
                "bbox_height": float(h),
                "bbox_area": float(w * h),
            }
        )
        perimeter = float(cv2.arcLength(biggest, True))
        result["compactness"] = float((4.0 * np.pi * area) / ((perimeter**2) + 1e-8))

    moments = cv2.moments(mask_binary)
    if moments["m00"] > 0:
        result["centroid_x"] = float(moments["m10"] / moments["m00"])
        result["centroid_y"] = float(moments["m01"] / moments["m00"])

    if np.any(mask_binary == 1):
        result["mask_mean_intensity_inside_tumor"] = float(gray_image[mask_binary == 1].mean())
    if np.any(mask_binary == 0):
        result["mask_mean_intensity_outside_tumor"] = float(gray_image[mask_binary == 0].mean())
    return result


def probability_overlay(rgb_image: np.ndarray, probability_mask: np.ndarray) -> np.ndarray:
    heat = np.uint8(np.clip(probability_mask, 0.0, 1.0) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_image, 0.65, heat, 0.35, 0)


def segmentation_overlay(rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    overlay = rgb_image.copy()
    if np.any(binary_mask == 1):
        overlay[binary_mask == 1] = (0.55 * overlay[binary_mask == 1] + 0.45 * np.array([72, 101, 255])).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 109, 57), 2)
    return overlay


class BrainTumorEngine:
    def __init__(self) -> None:
        self.ranges = load_numeric_ranges()

        self.classical_models: dict[str, Any] = {}
        classical_files = {
            "Random Forest (Deployed Best)": "best_classical_model.joblib",
            "Random Forest (Tuned)": "best_classical_model_tuned_best.joblib",
            "Gradient Boosting": "gradient_boosting.joblib",
            "Support Vector Machine (SVM)": "svm.joblib",
            "Decision Tree": "decision_tree.joblib",
            "Logistic Regression": "logistic_regression.joblib",
        }
        for label, fname in classical_files.items():
            fpath = BRAIN_MODELS_DIR / fname
            if fpath.exists():
                self.classical_models[label] = joblib.load(fpath)

        primary_classical = self.classical_models["Random Forest (Deployed Best)"]
        self.classical_features = list(primary_classical["features"])

        cnn_checkpoint = torch.load(BRAIN_MODELS_DIR / "cnn_classifier.pt", map_location=DEVICE, weights_only=False)
        self.class_names = list(cnn_checkpoint.get("class_names", BRAIN_CLASS_LABELS))
        self.classification_input_size = tuple(cnn_checkpoint.get("input_size", [224, 224]))
        self.classification_transform = transforms.Compose(
            [
                transforms.Resize(self.classification_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cnn_strategy_name = cnn_checkpoint.get("strategy_name", "efficientnet_b0_fold_ensemble")
        self.cnn_use_tta = bool(cnn_checkpoint.get("use_tta", False))
        self.ensemble_models = []
        for model_entry in cnn_checkpoint.get("models", []):
            model = build_classifier_model(model_entry["backbone_name"], len(self.class_names))
            model.load_state_dict(model_entry["state_dict"])
            model.to(DEVICE).eval()
            self.ensemble_models.append(
                {
                    "model": model,
                    "weight": float(model_entry.get("weight", 1.0)),
                    "backbone_name": model_entry.get("backbone_name", "efficientnet_b0"),
                    "run_name": model_entry.get("run_name", "fold"),
                }
            )

        self.single_backbones: dict[str, nn.Module] = {}
        for bname, fname in [
            ("efficientnet_b0", "cnn_efficientnet_b0.pt"),
            ("convnext_tiny", "cnn_convnext_tiny.pt"),
            ("resnet18", "cnn_resnet18.pt"),
        ]:
            bpath = BRAIN_MODELS_DIR / fname
            if bpath.exists():
                bck = torch.load(bpath, map_location=DEVICE, weights_only=False)
                bm = build_classifier_model(bname, len(self.class_names))
                bm.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in bck["state_dict"].items()})
                bm.to(DEVICE).eval()
                self.single_backbones[bname] = bm

        seg_path = BRAIN_MODELS_DIR / "seg_unet_small.pt"
        seg_ck = torch.load(seg_path, map_location=DEVICE, weights_only=False)
        self.segmentation_threshold = float(seg_ck.get("threshold", 0.35))
        self.segmentation_use_tta = bool(seg_ck.get("use_tta", True))
        self.segmentation_model_name = seg_ck.get("model_name", "unet_small")
        self.segmentation_input_size = tuple(seg_ck.get("input_size", [256, 256]))
        self.seg_model = UNetSmall().to(DEVICE)
        self.seg_model.load_state_dict(seg_ck["state_dict"])
        self.seg_model.eval()

    def classify(self, pil_image: Image.Image, backbone_choice: str = "ensemble", use_tta: bool = True) -> dict[str, Any]:
        tensor = self.classification_transform(pil_image).unsqueeze(0).to(DEVICE)

        backbone_comparison: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            total_w = sum(b["weight"] for b in self.ensemble_models)
            probs_accum = None
            for b in self.ensemble_models:
                p = classifier_forward_probs(b["model"], tensor, use_tta=use_tta)
                w_p = p * b["weight"]
                probs_accum = w_p if probs_accum is None else probs_accum + w_p
            ens_probs = (probs_accum / max(total_w, 1e-9)).cpu().numpy().ravel()
            ens_idx = int(np.argmax(ens_probs))
            backbone_comparison["ensemble"] = {
                "title": "EfficientNet-B0 (5-Fold Ensemble)",
                "predicted_class": self.class_names[ens_idx],
                "probability": round(float(ens_probs[ens_idx]), 4),
                "probabilities": {lbl: round(float(ens_probs[i]), 4) for i, lbl in enumerate(self.class_names)},
            }

            titles = {
                "efficientnet_b0": "EfficientNet-B0 (Single)",
                "convnext_tiny": "ConvNeXt-Tiny",
                "resnet18": "ResNet-18",
            }
            for bname, bm in self.single_backbones.items():
                bp = classifier_forward_probs(bm, tensor, use_tta=use_tta).cpu().numpy().ravel()
                b_idx = int(np.argmax(bp))
                backbone_comparison[bname] = {
                    "title": titles.get(bname, bname),
                    "predicted_class": self.class_names[b_idx],
                    "probability": round(float(bp[b_idx]), 4),
                    "probabilities": {lbl: round(float(bp[i]), 4) for i, lbl in enumerate(self.class_names)},
                }

        chosen = backbone_comparison.get(backbone_choice, backbone_comparison["ensemble"])
        return {
            "predicted_class": chosen["predicted_class"],
            "probability": chosen["probability"],
            "strategy_name": backbone_choice if backbone_choice != "ensemble" else self.cnn_strategy_name,
            "selected_model_title": chosen["title"],
            "tta_used": use_tta,
            "probabilities": chosen["probabilities"],
            "all_backbones": backbone_comparison,
        }

    def compute_saliency(self, pil_image: Image.Image) -> np.ndarray:
        tensor = self.classification_transform(pil_image).unsqueeze(0).to(DEVICE)
        model = self.ensemble_models[0]["model"]
        with torch.enable_grad():
            x = tensor.clone().requires_grad_(True)
            logits = model(x)
            target = logits.argmax(dim=1)
            logits[0, target[0]].backward()
            grad = x.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
        grad = cv2.GaussianBlur(grad, (11, 11), 0)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        return grad

    def segment(
        self, pil_image: Image.Image, original_shape: tuple[int, int], threshold: float | None = None
    ) -> dict[str, Any]:
        thresh = threshold if threshold is not None else self.segmentation_threshold
        resized = pil_image.resize(self.segmentation_input_size)
        image_arr = np.asarray(resized).astype(np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        tensor = torch.tensor(image_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.seg_model(tensor)
            if self.segmentation_use_tta:
                flipped = torch.flip(tensor, dims=[3])
                f_logits = torch.flip(self.seg_model(flipped), dims=[3])
                logits = (logits + f_logits) / 2.0
            probabilities = torch.sigmoid(logits).cpu().numpy()[0, 0]
        binary = (probabilities > thresh).astype(np.uint8)
        original_height, original_width = original_shape
        prob_up = cv2.resize(probabilities, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        mask_up = cv2.resize(binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return {
            "probability_mask": prob_up,
            "binary_mask": (mask_up > 0).astype(np.uint8),
            "threshold": thresh,
            "model_name": self.segmentation_model_name,
            "use_tta": self.segmentation_use_tta,
        }

    def classical_tumor_support(self, features: dict[str, float]) -> dict[str, Any]:
        feature_row = pd.DataFrame([{name: float(features[name]) for name in self.classical_features}])
        all_results = []
        for label, bundle in self.classical_models.items():
            pipe = bundle.get("pipeline")
            if pipe is not None:
                pred = int(pipe.predict(feature_row)[0])
                prob = float(pipe.predict_proba(feature_row)[0, 1]) if hasattr(pipe, "predict_proba") else 0.0
            else:
                m = bundle["model"]
                arr = feature_row.to_numpy(dtype=np.float32)
                pred = int(m.predict(arr)[0])
                prob = float(m.predict_proba(arr)[0, 1]) if hasattr(m, "predict_proba") else 0.0
            all_results.append(
                {
                    "model_name": label,
                    "predicted_binary_label": "tumor" if pred == 1 else "no_tumor",
                    "tumor_probability": round(prob, 4),
                }
            )
        primary = all_results[0]
        return {
            "predicted_binary_label": primary["predicted_binary_label"],
            "tumor_probability": primary["tumor_probability"],
            "model_name": primary["model_name"],
            "all_classical_models": all_results,
        }

    def analyze(
        self,
        pil_image: Image.Image,
        filename: str = "uploaded_mri.jpg",
        selected_plane: str = "auto",
        backbone_choice: str = "ensemble",
        segmentation_threshold: float = 0.35,
        use_tta: bool = True,
    ) -> dict[str, Any]:
        rgb_image = np.asarray(pil_image.convert("RGB"))
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        plane = parse_plane(filename, selected_plane)

        cnn_result = self.classify(pil_image, backbone_choice=backbone_choice, use_tta=use_tta)
        seg_result = self.segment(pil_image, gray_image.shape[:2], threshold=segmentation_threshold)
        saliency_map = self.compute_saliency(pil_image)
        saliency_up = cv2.resize(saliency_map, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_features = compute_mask_features(gray_image, seg_result["binary_mask"])

        mean_intensity = float(gray_image.mean())
        std_intensity = float(gray_image.std())
        entropy = shannon_entropy(gray_image)
        blur = blur_score(gray_image)
        edges = edge_density(gray_image)
        quality = (
            0.4 * normalize_value(blur, *self.ranges["blur_score"])
            + 0.3 * normalize_value(entropy, *self.ranges["image_entropy"])
            + 0.3 * normalize_value(std_intensity, *self.ranges["image_std_intensity"])
        ) * 100.0

        feature_dict = {
            "image_mean_intensity": mean_intensity,
            "image_std_intensity": std_intensity,
            "image_entropy": entropy,
            "blur_score": blur,
            "edge_density": edges,
            "image_quality_score": quality,
            "plane_code": float(PLANE_CODES.get(plane, 0)),
        }

        classical_result = self.classical_tumor_support(feature_dict)

        predicted_class = cnn_result["predicted_class"]
        tumor_area_percent = float(mask_features["tumor_area_ratio"] * 100.0)
        inside = mask_features.get("mask_mean_intensity_inside_tumor")
        outside = mask_features.get("mask_mean_intensity_outside_tumor")
        intensity_contrast = float(inside - outside) if inside is not None and outside is not None else 0.0
        area_component = normalize_value(tumor_area_percent, *self.ranges["tumor_area_percent"]) * 50.0
        contrast_component = normalize_value(intensity_contrast, *self.ranges["intensity_contrast"]) * 30.0
        class_component = float(CLASS_RISK_WEIGHTS.get(predicted_class, 8))
        score = float(np.clip(area_component + contrast_component + class_component, 0.0, 100.0))

        if predicted_class == "no_tumor":
            score = min(score, 18.0)

        if tumor_area_percent <= 0.0 or predicted_class == "no_tumor":
            size_category = "none"
        elif tumor_area_percent <= 1.5:
            size_category = "small"
        elif tumor_area_percent <= 5.0:
            size_category = "medium"
        else:
            size_category = "large"

        if score <= 33.33:
            risk_level = "low"
            priority = "Routine neuroimaging review"
        elif score <= 66.66:
            risk_level = "medium"
            priority = "Specialist neuroradiology review recommended"
        else:
            risk_level = "high"
            priority = "Urgent neuro-oncology & neurosurgery review recommended"

        explanation = (
            f"Combined predicted pathology ({predicted_class.replace('_', ' ')}), segmentation-estimated lesion "
            f"burden ({tumor_area_percent:.2f}% area, {size_category}), and intensity contrast ({intensity_contrast:.1f}) "
            f"to compute a clinical triage score of {score:.1f}/100 ({risk_level} priority)."
        )

        overlay = segmentation_overlay(rgb_image, seg_result["binary_mask"])
        probability_view = probability_overlay(rgb_image, seg_result["probability_mask"])
        saliency_view = probability_overlay(rgb_image, saliency_up)
        binary_mask_view = np.uint8(seg_result["binary_mask"] * 255)

        return {
            "modality": "brain_mri",
            "filename": filename,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "image_size": [rgb_image.shape[1], rgb_image.shape[0]],
            "plane": plane,
            "device": str(DEVICE),
            "finding": "No tumor detected" if predicted_class == "no_tumor" else f"Brain tumor detected ({predicted_class.capitalize()})",
            "classification": cnn_result,
            "classical_support": classical_result,
            "segmentation": {
                "model_name": seg_result["model_name"],
                "threshold": round(seg_result["threshold"], 2),
                "use_tta": seg_result["use_tta"],
                "tumor_area_px": round(mask_features["tumor_area_px"], 1),
                "tumor_area_percent": round(tumor_area_percent, 2),
                "bbox_area": round(mask_features["bbox_area"], 1),
                "centroid": [
                    round(mask_features["centroid_x"], 1) if mask_features["centroid_x"] is not None else None,
                    round(mask_features["centroid_y"], 1) if mask_features["centroid_y"] is not None else None,
                ],
                "compactness": None if mask_features["compactness"] is None else round(mask_features["compactness"], 4),
            },
            "risk_summary": {
                "tumor_type": predicted_class,
                "tumor_area_percent": round(tumor_area_percent, 2),
                "tumor_size_category": size_category,
                "intensity_contrast": round(intensity_contrast, 2),
                "risk_score": round(score, 1),
                "risk_level": risk_level,
                "suggested_priority": priority,
                "explanation": explanation,
                "disclaimer": "Research prototype only. Not a clinical diagnosis.",
            },
            "engineered_features": {key: round(float(val), 4) for key, val in feature_dict.items()},
            "preview_images": {
                "original": image_to_base64(rgb_image),
                "overlay": image_to_base64(overlay),
                "probability": image_to_base64(probability_view),
                "saliency": image_to_base64(saliency_view),
                "binary_mask": image_to_base64(binary_mask_view, mode="gray"),
            },
        }
