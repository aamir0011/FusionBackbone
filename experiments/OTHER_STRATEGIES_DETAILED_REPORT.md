# Detailed Report: Other Missing-Modality Strategies (Excluding Gating / Gating+Projection)

This report compiles implementation details and quantitative results for all other implemented strategies in my workspace. Baseline reference is `exp/vqarad-pathvqa-run1` (IRENE).

## 1) Strategy implementation details

| Strategy | Branch | Main training script | Core implementation idea | Training objective |
|---|---|---|---|---|
| cross_modal_distillation | `feat/mm-cross-modal-distillation` | `IRENE/train_cross_modal_distill.py` | Teacher (full modalities) + student (missing modalities) distillation | BCE + alpha*T^2*KL(teacher||student) |
| modality_dropout | `feat/mm-modality-dropout` | `IRENE/train_modality_dropout.py` | Stochastic modality dropout (image/text and optional structured) with keep masks | BCE classification |
| hemis_latent | `feat/mm-hemis-latent-aggregation` | `IRENE/train_hemis_latent.py` | HeMIS-inspired latent aggregation with stochastic masking curriculum | Model loss under masked modalities (branch-specific) |
| mmp_projection | `feat/mm-masked-modality-reconstruction` | `IRENE/train_masked_modality_recon.py` | Masked modality projection/reconstruction (MMP-style) | BCE + projection/reconstruction term |
| masked_modality_training | `feat/masked-modality-training` | `IRENE/train_irene_masked.py` | Random modality masking during training; masked-aware IRENE variant | BCE classification on masked samples |
| smil_style | `feat/mm-smil-severe-missingness` | `IRENE/train_smil_style.py` | Severe missingness schedule + consistency (SMIL-style) | BCE + meta consistency KL |
| set_perceiver_fusion | `feat/mm-set-perceiver-fusion` | `IRENE/train_set_perceiver_fusion.py` | Set/Perceiver-style modality-agnostic token fusion with availability embedding | BCE classification |
| single_branch_invariant | `feat/mm-single-branch-invariant` | `IRENE/train_single_branch_invariant.py` | Single shared branch with full-vs-masked consistency | BCE + cons_w*KL consistency |
| generative_imputation | `feat/mm-generative-imputation` | `IRENE/train_generative_imputation.py` | Text-feature imputer reconstructs missing text embeddings | BCE + impute_w*MSE(recon,text) |
| missing_tokens_presence_masks | `feat/mm-missing-tokens-presence-mask` | `IRENE/train_missing_tokens_presence.py` | Learned missing tokens + explicit presence-mask projection | BCE classification |
| uncertainty_aware_fusion | `feat/mm-uncertainty-aware-fusion` | `IRENE/train_uncertainty_fusion.py` | Multi-head logits + uncertainty-weighted fusion | BCE + uncertainty regularizer |

## 2) Dataset-level performance vs baseline

| Strategy | Dataset | Technique AUC | Baseline AUC | Delta (tech - baseline) |
|---|---|---:|---:|---:|
| cross_modal_distillation | vqarad | 0.9796 | 0.9867 | -0.0071 |
| cross_modal_distillation | pathvqa | 0.7712 | 0.8168 | -0.0456 |
| cross_modal_distillation | mimic | 0.6751 | 0.7382 | -0.0631 |
| cross_modal_distillation | rexgradient | 0.8837 | 0.8948 | -0.0111 |
| generative_imputation | vqarad | 0.8326 | 0.9867 | -0.1541 |
| generative_imputation | pathvqa | 0.4980 | 0.8168 | -0.3188 |
| generative_imputation | mimic | 0.5667 | 0.7382 | -0.1715 |
| generative_imputation | rexgradient | 0.6515 | 0.8948 | -0.2433 |
| hemis_latent | vqarad | 0.9762 | 0.9867 | -0.0105 |
| hemis_latent | pathvqa | 0.6849 | 0.8168 | -0.1326 |
| hemis_latent | mimic | 0.7152 | 0.7382 | -0.0229 |
| hemis_latent | rexgradient | 0.8919 | 0.8948 | -0.0029 |
| masked_modality_training | vqarad | 0.9139 | 0.9867 | -0.0728 |
| masked_modality_training | pathvqa | 0.5518 | 0.8168 | -0.2650 |
| masked_modality_training | mimic | 0.7130 | 0.7382 | -0.0252 |
| masked_modality_training | rexgradient | 0.7935 | 0.8948 | -0.1013 |
| missing_tokens_presence_masks | vqarad | 0.8684 | 0.9867 | -0.1183 |
| missing_tokens_presence_masks | pathvqa | 0.4902 | 0.8168 | -0.3266 |
| missing_tokens_presence_masks | mimic | 0.5357 | 0.7382 | -0.2025 |
| missing_tokens_presence_masks | rexgradient | 0.6539 | 0.8948 | -0.2409 |
| mmp_projection | vqarad | 0.8633 | 0.9867 | -0.1234 |
| mmp_projection | pathvqa | 0.7544 | 0.8168 | -0.0631 |
| mmp_projection | mimic | 0.7555 | 0.7382 | +0.0174 |
| mmp_projection | rexgradient | 0.8270 | 0.8948 | -0.0678 |
| modality_dropout | vqarad | 0.9799 | 0.9867 | -0.0068 |
| modality_dropout | pathvqa | 0.7718 | 0.8168 | -0.0450 |
| modality_dropout | mimic | 0.6974 | 0.7382 | -0.0408 |
| modality_dropout | rexgradient | 0.8247 | 0.8948 | -0.0701 |
| set_perceiver_fusion | vqarad | 0.5261 | 0.9867 | -0.4606 |
| set_perceiver_fusion | pathvqa | 0.6414 | 0.8168 | -0.1754 |
| set_perceiver_fusion | mimic | 0.6786 | 0.7382 | -0.0596 |
| set_perceiver_fusion | rexgradient | 0.7967 | 0.8948 | -0.0981 |
| single_branch_invariant | vqarad | 0.8530 | 0.9867 | -0.1337 |
| single_branch_invariant | pathvqa | 0.5421 | 0.8168 | -0.2754 |
| single_branch_invariant | mimic | 0.5184 | 0.7382 | -0.2197 |
| single_branch_invariant | rexgradient | 0.6600 | 0.8948 | -0.2348 |
| smil_style | vqarad | 0.8641 | 0.9867 | -0.1226 |
| smil_style | pathvqa | 0.6011 | 0.8168 | -0.2164 |
| smil_style | mimic | 0.6631 | 0.7382 | -0.0750 |
| smil_style | rexgradient | 0.8042 | 0.8948 | -0.0907 |
| uncertainty_aware_fusion | vqarad | 0.2318 | 0.9867 | -0.7549 |
| uncertainty_aware_fusion | pathvqa | 0.3879 | 0.8168 | -0.4289 |
| uncertainty_aware_fusion | mimic | 0.6375 | 0.7382 | -0.1007 |
| uncertainty_aware_fusion | rexgradient | 0.6381 | 0.8948 | -0.2567 |

## 3) Average delta ranking across 4 datasets

| Rank | Strategy | Avg Delta vs Baseline |
|---:|---|---:|
| 1 | cross_modal_distillation | -0.0317 |
| 2 | modality_dropout | -0.0407 |
| 3 | hemis_latent | -0.0422 |
| 4 | mmp_projection | -0.0592 |
| 5 | masked_modality_training | -0.1161 |
| 6 | smil_style | -0.1262 |
| 7 | set_perceiver_fusion | -0.1984 |
| 8 | single_branch_invariant | -0.2159 |
| 9 | generative_imputation | -0.2219 |
| 10 | missing_tokens_presence_masks | -0.2221 |
| 11 | uncertainty_aware_fusion | -0.3853 |

## 4) Disease-wise comparison (available techniques)

Disease-wise deltas are available from `experiments/new_techniques_compare/comparison_per_disease.csv` for: HeMIS-latent, Single-branch invariant, SMIL-style, and MMP-projection.

### VQARAD disease-wise deltas
| Disease | hemis_latent Δ | single_branch_invariant Δ | smil_style Δ | mmp_projection Δ |
|---|---:|---:|---:|---:|
| Bronchiectasis | +nan | +nan | +nan | +nan |
| COPD | +nan | +nan | +nan | +nan |
| ILD | +nan | +nan | +nan | +nan |
| Lung cancer | +nan | +nan | +nan | +nan |
| Pleural effusion | +0.0000 | -0.0728 | -0.0592 | -0.0728 |
| Pneumonia | -0.0043 | -0.1744 | -0.1546 | -0.1469 |
| Pneumothorax | -0.0271 | -0.1540 | -0.1540 | -0.1506 |
| Tuberculosis | +nan | +nan | +nan | +nan |

### PATHVQA disease-wise deltas
| Disease | hemis_latent Δ | single_branch_invariant Δ | smil_style Δ | mmp_projection Δ |
|---|---:|---:|---:|---:|
| Bronchiectasis | +nan | +nan | +nan | +nan |
| COPD | +nan | +nan | +nan | +nan |
| ILD | -0.3903 | -0.2750 | -0.3795 | -0.0410 |
| Lung cancer | -0.0323 | -0.4408 | -0.2665 | -0.1187 |
| Pleural effusion | +nan | +nan | +nan | +nan |
| Pneumonia | -0.0166 | -0.4424 | -0.2206 | -0.0845 |
| Pneumothorax | +nan | +nan | +nan | +nan |
| Tuberculosis | -0.0914 | +0.0566 | +0.0010 | -0.0083 |

### MIMIC disease-wise deltas
| Disease | hemis_latent Δ | single_branch_invariant Δ | smil_style Δ | mmp_projection Δ |
|---|---:|---:|---:|---:|
| Bronchiectasis | -0.1066 | -0.1553 | +0.0820 | -0.0675 |
| COPD | +0.1036 | -0.2486 | -0.2291 | +0.0084 |
| ILD | +0.0041 | -0.3746 | -0.0332 | -0.0009 |
| Lung cancer | -0.0599 | -0.2684 | -0.2287 | -0.0597 |
| Pleural effusion | +0.0071 | -0.4845 | -0.1112 | -0.0173 |
| Pneumonia | -0.0228 | -0.2778 | -0.2490 | -0.1545 |
| Pneumothorax | -0.0095 | -0.3670 | -0.1494 | -0.0217 |
| Tuberculosis | -0.0990 | +0.4187 | +0.3187 | +0.4522 |

### REXGRADIENT disease-wise deltas
| Disease | hemis_latent Δ | single_branch_invariant Δ | smil_style Δ | mmp_projection Δ |
|---|---:|---:|---:|---:|
| Bronchiectasis | -0.1351 | -0.3211 | -0.1864 | -0.3383 |
| COPD | +0.0632 | -0.1144 | -0.1896 | +0.0007 |
| ILD | -0.0002 | -0.3199 | -0.0134 | +0.0011 |
| Lung cancer | -0.0382 | -0.0471 | -0.1969 | -0.0952 |
| Pleural effusion | +0.0066 | -0.3927 | -0.0262 | -0.0040 |
| Pneumonia | +0.0043 | -0.3706 | -0.1252 | -0.0976 |
| Pneumothorax | +0.0048 | -0.4296 | -0.0255 | -0.0106 |
| Tuberculosis | +0.0712 | +0.1167 | +0.0380 | +0.0013 |

## 5) Interpretation for thesis and paper

- Among non-gating methods, **cross_modal_distillation** and **modality_dropout** show the smallest average degradation vs baseline.
- **HeMIS-latent** is the strongest among newly proposed variants, with near-baseline performance on ReXGradient and mild drop on MIMIC.
- **Single-branch invariant** and **SMIL-style severe missingness** underperform consistently in these runs.
- **MMP projection** improves MIMIC overall but is unstable across VQA-RAD/PathVQA/ReXGradient.
- For thesis framing: present these methods as ablations that map the robustness-accuracy frontier; emphasize that not all missing-modality mechanisms generalize uniformly across datasets.