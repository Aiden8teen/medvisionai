# AI-Assisted Brain Tumor Treatment Planning from MRI

## 1. Study Positioning
This version of the notebook is structured as a comparative study: it documents preprocessing changes, baseline-vs-tuned classical models, advanced deep learning backbones, ensemble/TTA inference, and no-regression deployment logic.

## 2. Output Folders
- Main graphs: `/kaggle/working/graphs`
- Main reports: `/kaggle/working/reports`
- Comparative-study images: `/kaggle/working/comparative_study/images`
- Comparative-study tables: `/kaggle/working/comparative_study/tables`
- Comparative-study markdown reports: `/kaggle/working/comparative_study/reports`

## 3. Run Configuration
Notebook version: `brisc-kaggle-v7-performance-max`
FAST_MODE: `False`
Device used: `cuda`
CNN backbones: `['resnet18', 'efficientnet_b0', 'convnext_tiny']`
Segmentation models: `['unet_small', 'deeplabv3_resnet50']`

## 4. Dataset Description
Detected dataset path: `/kaggle/input/datasets/briscdataset/brisc2025/brisc2025`
Images processed: 10,793

## 5. Classical ML Comparative Results
Best baseline F1: 0.9823468328141225
Best tuned F1: 0.9818370524130773
Deployed classical F1: 0.9823468328141225
Best classical summary: {'model': 'Random Forest', 'f1': np.float64(0.9823468328141225), 'auc': np.float64(0.9930663424469646), 'cv_f1': np.float64(0.9863264755454815), 'selection_source': 'baseline'}

## 6. CNN Results
CNN summary: {'trained': True, 'model_name': 'efficientnet_b0', 'strategy_name': 'efficientnet_b0_fold_ensemble', 'backbones': '["efficientnet_b0", "efficientnet_b0", "efficientnet_b0", "efficientnet_b0", "efficientnet_b0"]', 'tta_used': False, 'loss_name': 'weighted_focal_loss', 'accuracy': 0.993, 'f1': 0.9936006989274982, 'eval_split': 'test', 'best_epoch': 6, 'best_val_f1': 0.9932442775087593, 'epochs_ran': 6, 'early_stopped': True}

## 7. Segmentation Results
Segmentation summary: {'trained': True, 'model_name': 'deeplabv3_resnet50', 'dice': 0.7927802267462708, 'iou': 0.6918209564893745, 'pixel_accuracy': 0.9895016337549963, 'threshold': 0.5, 'use_tta': True, 'setting': 'default_tta_0.50', 'default_threshold_dice': 0.77679067877836, 'best_epoch': 6, 'best_val_dice': 0.7828310268179415, 'epochs_ran': 8}
Deployment Dice gain vs default no-TTA 0.50: 0.015990

## 8. Saved Figures
- `/kaggle/working/graphs/class_distribution.png`
- `/kaggle/working/graphs/classical_baseline_vs_tuned_f1.png`
- `/kaggle/working/graphs/classical_cv_f1_boxplot.png`
- `/kaggle/working/graphs/cnn_confusion_matrix.png`
- `/kaggle/working/graphs/cnn_training_curves.png`
- `/kaggle/working/graphs/confusion_matrices.png`
- `/kaggle/working/graphs/correlation_heatmap_triangular.png`
- `/kaggle/working/graphs/filter_feature_scores.png`
- `/kaggle/working/graphs/gradient_boosting_feature_importance.png`
- `/kaggle/working/graphs/image_intensity_by_class_boxplot.png`
- `/kaggle/working/graphs/model_comparison_barplot.png`
- `/kaggle/working/graphs/pairplot_top_features.png`
- `/kaggle/working/graphs/permutation_importance.png`
- `/kaggle/working/graphs/plane_distribution.png`
- `/kaggle/working/graphs/preprocessing_comparison_grid.png`
- `/kaggle/working/graphs/random_forest_feature_importance.png`
- `/kaggle/working/graphs/rfe_score_by_number_features.png`
- `/kaggle/working/graphs/risk_level_distribution.png`
- `/kaggle/working/graphs/roc_curves.png`
- `/kaggle/working/graphs/saliency_examples.png`
- `/kaggle/working/graphs/sample_mask_overlay_grid.png`
- `/kaggle/working/graphs/sample_mri_grid.png`
- `/kaggle/working/graphs/scatter_tumor_area_vs_intensity_contrast.png`
- `/kaggle/working/graphs/segmentation_examples.png`
- `/kaggle/working/graphs/segmentation_preprocessing_grid.png`
- `/kaggle/working/graphs/segmentation_threshold_before_after.png`
- `/kaggle/working/graphs/segmentation_threshold_sweep.png`
- `/kaggle/working/graphs/segmentation_training_curves.png`
- `/kaggle/working/graphs/target_correlation_heatmap.png`
- `/kaggle/working/graphs/top_10_correlations.png`
- `/kaggle/working/graphs/train_test_distribution.png`
- `/kaggle/working/graphs/tumor_area_by_class_boxplot.png`
- `/kaggle/working/graphs/tumor_area_distribution.png`

## 9. Comparative Study Notes
Detailed methodology, technique benefits, and omission risks are written to `comparative_study_methodology.md` and mirrored into the comparative-study folder.

## 10. Conclusion
This notebook now combines stronger deep learning architectures, weighted focal losses, fold-ensemble classification, pretrained segmentation comparison, and no-regression deployment safeguards to maximize delivered performance.