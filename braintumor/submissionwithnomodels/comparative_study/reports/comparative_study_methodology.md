# Comparative Study Methodology

## Step-by-Step Workflow
1. Detect the BRISC dataset and build the shared metadata table.
2. Clean duplicates, invalid images, and mask mismatches while keeping medically plausible extremes.
3. Engineer tumor, geometry, intensity, and risk-oriented features.
4. Save before/after preprocessing figures for raw MRI, enhanced MRI, and segmentation overlays.
5. Prepare the classical ML dataset with train/test separation and feature selection.
6. Run stratified cross-validation for all baseline classical models.
7. Tune the top classical models but keep the stronger held-out artifact if tuning regresses.
8. Benchmark multiple CNN backbones, apply weighted focal loss, and compare single, TTA, and fold-ensemble strategies.
9. Benchmark multiple segmentation architectures, apply focal-enhanced loss, and compare threshold/TTA deployment settings.
10. Export images, CSV tables, markdown notes, and model files into a structured comparative-study folder.

## Techniques, Benefits, and Omission Risks
| Technique | Used in | Purpose | Benefit | If omitted |
| --- | --- | --- | --- | --- |
| Preprocessing before/after grids | EDA and reporting | Visualize how CLAHE, denoising, edge maps, and mask overlays change the raw MRI before modeling. | Creates report-ready evidence for data quality and preprocessing impact. | Your report loses visual proof of what the pipeline changed before training. |
| No-regression artifact retention | Classical ML and segmentation | Keep the strongest deployed artifact even when a more advanced variant underperforms. | Maximizes delivered performance. | A weaker tuned model can replace a stronger baseline. |
| Multiple pretrained CNN backbones | CNN classification | Benchmark multiple architectures instead of relying on one model family. | Improves the chance of finding a better feature extractor. | A stronger backbone may be missed. |
| Weighted focal loss | CNN classification and segmentation | Focus learning on harder or underrepresented examples. | Often improves robustness and class balance behavior. | The model can under-focus on difficult cases. |
| Fold ensemble | CNN classification | Average multiple independently trained fold members. | Reduces variance and can recover small performance gains. | Single-model noise can remain in final inference. |
| Weighted ensemble and TTA | CNN classification and segmentation | Average predictions across models and augmented views. | Improves robustness without changing labels or data collection. | Final predictions may be less stable. |
| Pretrained encoder segmentation model | Segmentation | Compare the handcrafted U-Net with a stronger pretrained architecture. | Can improve mask quality through richer image features. | Segmentation performance may stay limited by a smaller architecture. |

## Result Highlights
- Best baseline classical F1: `0.9823468328141225`
- Best tuned classical F1: `0.9818370524130773`
- Deployed classical F1: `0.9823468328141225` from `baseline`
- CNN strategy: `efficientnet_b0_fold_ensemble` with macro F1 `0.9936006989274982`
- Segmentation model/setting: `deeplabv3_resnet50 / default_tta_0.50` with Dice `0.7927802267462708`
- Deployment Dice gain over default no-TTA 0.50: `0.015989547967910744`

## Report-Ready Files
- Before/after figures are mirrored into `/kaggle/working/comparative_study/images/before_after`
- Classical-study tables are mirrored into `/kaggle/working/comparative_study/tables`
- Markdown summaries are mirrored into `/kaggle/working/comparative_study/reports`

## Why This Version Is Stronger
The notebook is now a performance-first study package: it benchmarks deeper backbones, uses focal losses, supports fold ensembles, compares pretrained segmentation models, and preserves stronger deployment choices when more complex variants fail to improve the final result.