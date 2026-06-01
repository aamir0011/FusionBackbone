# Comprehensive Report: Missing-Modality Experiments

## Baseline Reference

Baseline paper implementation (`exp/vqarad-pathvqa-run1`, 10 epochs):

| Dataset | Baseline AUC |
|---|---:|
| vqarad | 0.9867 |
| pathvqa | 0.8168 |
| mimic | 0.7382 |
| rexgradient | 0.8948 |

## Overall Ranking (Average Delta vs Baseline, 4-dataset runs)

| Rank | Technique | Variant | Avg Delta |
|---:|---|---|---:|
| 1 | Phase A: ShaSpec + Gating | 50ep fullscale | +0.0391 |
| 2 | Gating | 50ep fullscale | +0.0034 |
| 3 | Modality-wise Gating | 10ep fullscale | -0.0256 |
| 4 | Cross-modal Distillation | 10ep fullscale | -0.0317 |
| 5 | Modality Dropout | 10ep fullscale | -0.0407 |
| 6 | Hemis Latent | new-techniques compare | -0.0422 |
| 7 | Mmp Projection | new-techniques compare | -0.0592 |
| 8 | Masked Modality Training | 10ep fullscale | -0.1161 |
| 9 | Smil Style | new-techniques compare | -0.1262 |
| 10 | Gating + Projection | 10ep fullscale | -0.1371 |
| 11 | Set/Perceiver Fusion | 10ep fullscale | -0.1984 |
| 12 | Single Branch Invariant | new-techniques compare | -0.2159 |
| 13 | Generative Imputation | 10ep fullscale | -0.2219 |
| 14 | Missing Tokens + Presence Masks | 10ep fullscale | -0.2221 |
| 15 | Uncertainty-aware Fusion | 10ep fullscale | -0.3853 |

## 1. Masked Modality Training (10ep fullscale)

### 1) Introduction
Trains with random modality masking so the model learns from partial multimodal inputs instead of assuming complete inputs.

### 2) Methodology (implementation w.r.t. baseline)
Implements a masked-aware IRENE training path (train_irene_masked.py) where modality streams are stochastically masked during training while retaining the baseline classifier objective.
Source implementation/result file: `experiments/fullscale_masked-modality-training_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9139 | 0.9867 | -0.0728 |
| pathvqa | 0.5518 | 0.8168 | -0.2650 |
| mimic | 0.7130 | 0.7382 | -0.0252 |
| rexgradient | 0.7935 | 0.8948 | -0.1013 |

- Average delta vs baseline: **-0.1161**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 2. Modality Dropout (10ep fullscale)

### 1) Introduction
A regularization strategy that randomly drops modalities during training to improve robustness to missing inputs.

### 2) Methodology (implementation w.r.t. baseline)
Keeps baseline architecture and applies Bernoulli keep masks to image/text (and optionally structured) streams before fusion (train_modality_dropout.py).
Source implementation/result file: `experiments/fullscale_mm-modality-dropout_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9799 | 0.9867 | -0.0068 |
| pathvqa | 0.7718 | 0.8168 | -0.0450 |
| mimic | 0.6974 | 0.7382 | -0.0408 |
| rexgradient | 0.8247 | 0.8948 | -0.0701 |

- Average delta vs baseline: **-0.0407**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 3. Modality-wise Gating (10ep fullscale)

### 1) Introduction
Learns per-modality reliability weights to adaptively scale each modality according to availability.

### 2) Methodology (implementation w.r.t. baseline)
Adds a small gating MLP over modality-presence indicators; gated modality features are passed to baseline IRENE classifier (train_modality_gating.py).
Source implementation/result file: `experiments/fullscale_mm-modality-wise-gating_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9856 | 0.9867 | -0.0011 |
| pathvqa | 0.7795 | 0.8168 | -0.0373 |
| mimic | 0.6735 | 0.7382 | -0.0647 |
| rexgradient | 0.8956 | 0.8948 | +0.0008 |

- Average delta vs baseline: **-0.0256**

### 5) Comparison with baseline (which is better and why)
- Verdict: **better on 1/4 datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 4. Set/Perceiver Fusion (10ep fullscale)

### 1) Introduction
Uses modality-agnostic token fusion via set/perceiver style latent bottlenecks.

### 2) Methodology (implementation w.r.t. baseline)
Replaces baseline fusion path with a set/perceiver fusion module that ingests available modality tokens and predicts labels (train_set_perceiver_fusion.py).
Source implementation/result file: `experiments/fullscale_mm-set-perceiver-fusion_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.5261 | 0.9867 | -0.4606 |
| pathvqa | 0.6414 | 0.8168 | -0.1754 |
| mimic | 0.6786 | 0.7382 | -0.0596 |
| rexgradient | 0.7967 | 0.8948 | -0.0981 |

- Average delta vs baseline: **-0.1984**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 5. Cross-modal Distillation (10ep fullscale)

### 1) Introduction
Transfers knowledge from a full-modality teacher to a missing-modality student.

### 2) Methodology (implementation w.r.t. baseline)
Trains teacher and student; student sees masked modalities and optimizes BCE + KL distillation from teacher logits (train_cross_modal_distill.py).
Source implementation/result file: `experiments/fullscale_mm-cross-modal-distillation_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9796 | 0.9867 | -0.0071 |
| pathvqa | 0.7712 | 0.8168 | -0.0456 |
| mimic | 0.6751 | 0.7382 | -0.0631 |
| rexgradient | 0.8837 | 0.8948 | -0.0111 |

- Average delta vs baseline: **-0.0317**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 6. Generative Imputation (10ep fullscale)

### 1) Introduction
Imputes missing modality representations (especially text) before classification.

### 2) Methodology (implementation w.r.t. baseline)
Adds imputer network that reconstructs modality embeddings; optimizes BCE + MSE reconstruction loss (train_generative_imputation.py).
Source implementation/result file: `experiments/fullscale_mm-generative-imputation_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8326 | 0.9867 | -0.1541 |
| pathvqa | 0.4980 | 0.8168 | -0.3188 |
| mimic | 0.5667 | 0.7382 | -0.1715 |
| rexgradient | 0.6515 | 0.8948 | -0.2433 |

- Average delta vs baseline: **-0.2219**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 7. Missing Tokens + Presence Masks (10ep fullscale)

### 1) Introduction
Handles missingness using learned missing tokens and explicit presence-mask embeddings.

### 2) Methodology (implementation w.r.t. baseline)
Injects learned missing tokens and projected presence masks into modality streams, then uses baseline BCE classification (train_missing_tokens_presence.py).
Source implementation/result file: `experiments/fullscale_mm-missing-tokens-presence-mask_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8684 | 0.9867 | -0.1183 |
| pathvqa | 0.4902 | 0.8168 | -0.3266 |
| mimic | 0.5357 | 0.7382 | -0.2025 |
| rexgradient | 0.6539 | 0.8948 | -0.2409 |

- Average delta vs baseline: **-0.2221**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 8. Uncertainty-aware Fusion (10ep fullscale)

### 1) Introduction
Fuses modality predictions using uncertainty estimates to reduce overconfident errors under missingness.

### 2) Methodology (implementation w.r.t. baseline)
Adds uncertainty-aware fusion heads and regularization term on variance/calibration with BCE objective (train_uncertainty_fusion.py).
Source implementation/result file: `experiments/fullscale_mm-uncertainty-aware-fusion_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.2318 | 0.9867 | -0.7549 |
| pathvqa | 0.3879 | 0.8168 | -0.4289 |
| mimic | 0.6375 | 0.7382 | -0.1007 |
| rexgradient | 0.6381 | 0.8948 | -0.2567 |

- Average delta vs baseline: **-0.3853**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 9. Hemis Latent (new-techniques compare)

### 1) Introduction
HeMIS-inspired latent statistical aggregation for arbitrary modality subsets.

### 2) Methodology (implementation w.r.t. baseline)
Uses latent aggregation over available modality embeddings with masking curriculum (train_hemis_latent.py).
Source implementation/result file: `experiments/new_techniques_compare/comparison_summary.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9762 | 0.9867 | -0.0105 |
| pathvqa | 0.6849 | 0.8168 | -0.1326 |
| mimic | 0.7152 | 0.7382 | -0.0229 |
| rexgradient | 0.8919 | 0.8948 | -0.0029 |

- Average delta vs baseline: **-0.0422**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 10. Single Branch Invariant (new-techniques compare)

### 1) Introduction
Enforces invariant features between full and masked passes in a single branch.

### 2) Methodology (implementation w.r.t. baseline)
Adds consistency KL between full and masked forward passes with shared projection (train_single_branch_invariant.py).
Source implementation/result file: `experiments/new_techniques_compare/comparison_summary.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8530 | 0.9867 | -0.1337 |
| pathvqa | 0.5421 | 0.8168 | -0.2754 |
| mimic | 0.5184 | 0.7382 | -0.2197 |
| rexgradient | 0.6600 | 0.8948 | -0.2348 |

- Average delta vs baseline: **-0.2159**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 11. Smil Style (new-techniques compare)

### 1) Introduction
Severe-missingness curriculum/meta-consistency training inspired by SMIL setting.

### 2) Methodology (implementation w.r.t. baseline)
Uses high missingness schedules and consistency regularization beyond baseline BCE (train_smil_style.py).
Source implementation/result file: `experiments/new_techniques_compare/comparison_summary.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8641 | 0.9867 | -0.1226 |
| pathvqa | 0.6011 | 0.8168 | -0.2164 |
| mimic | 0.6631 | 0.7382 | -0.0750 |
| rexgradient | 0.8042 | 0.8948 | -0.0907 |

- Average delta vs baseline: **-0.1262**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 12. Mmp Projection (new-techniques compare)

### 1) Introduction
Masked modality projection/reconstruction to estimate unavailable modality content.

### 2) Methodology (implementation w.r.t. baseline)
Adds projection/reconstruction branch and auxiliary loss to recover masked modality signals (train_masked_modality_recon.py).
Source implementation/result file: `experiments/new_techniques_compare/comparison_summary.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8633 | 0.9867 | -0.1234 |
| pathvqa | 0.7544 | 0.8168 | -0.0631 |
| mimic | 0.7555 | 0.7382 | +0.0174 |
| rexgradient | 0.8270 | 0.8948 | -0.0678 |

- Average delta vs baseline: **-0.0592**

### 5) Comparison with baseline (which is better and why)
- Verdict: **better on 1/4 datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 13. Gating (50ep fullscale)

### 1) Introduction
Extended training of gating to test whether longer optimization closes baseline gap.

### 2) Methodology (implementation w.r.t. baseline)
Same gating mechanism as 10ep but trained 50 epochs on complete datasets.
Source implementation/result file: `experiments/fullscale_gating_50ep/fullscale_gating_50ep_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9537 | 0.9867 | -0.0330 |
| pathvqa | 0.8491 | 0.8168 | +0.0323 |
| mimic | 0.7779 | 0.7382 | +0.0397 |
| rexgradient | 0.8695 | 0.8948 | -0.0253 |

- Average delta vs baseline: **+0.0034**

### 5) Comparison with baseline (which is better and why)
- Verdict: **better on 2/4 datasets**.
- Mixed result: gains are dataset-dependent; likely sensitive to modality distribution and training dynamics.

## 14. Gating + Projection (10ep fullscale)

### 1) Introduction
Combines gating with learned modality projection for missing modalities.

### 2) Methodology (implementation w.r.t. baseline)
Implements gated blending between real and projected modality features with auxiliary projection loss (train_gating_variants.py --variant gating_proj).
Source implementation/result file: `experiments/fullscale_gating_proj_plan4/fullscale_gating_proj_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.8732 | 0.9867 | -0.1135 |
| pathvqa | 0.5485 | 0.8168 | -0.2683 |
| mimic | 0.6760 | 0.7382 | -0.0622 |
| rexgradient | 0.7902 | 0.8948 | -0.1046 |

- Average delta vs baseline: **-0.1371**

### 5) Comparison with baseline (which is better and why)
- Verdict: **worse than baseline on all datasets**.
- Why worse: robustness mechanism likely introduced optimization or representation trade-offs that reduced full-condition AUC.

## 15. Phase A: ShaSpec + Gating (50ep fullscale)

### 1) Introduction
Combines gating with shared-specific representation learning to preserve task signal under missingness.

### 2) Methodology (implementation w.r.t. baseline)
Adds shared/specific heads, alignment loss, domain loss, and orthogonality regularization on top of gating backbone (train_shaspec_gating.py).
Source implementation/result file: `experiments/phaseA_shaspec_gating_50ep/phaseA_shaspec_gating_50ep_all_datasets.csv`

### 3) Experiments conducted
- Number of experiment runs: **4**
- Datasets covered in this run: vqarad, pathvqa, mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| vqarad | 0.9760 | 0.9867 | -0.0107 |
| pathvqa | 0.9126 | 0.8168 | +0.0958 |
| mimic | 0.7975 | 0.7382 | +0.0593 |
| rexgradient | 0.9070 | 0.8948 | +0.0122 |

- Average delta vs baseline: **+0.0391**

### 5) Comparison with baseline (which is better and why)
- Verdict: **better on 3/4 datasets**.
- Why better: the technique likely improves robustness to modality incompleteness while preserving discriminative multimodal information.

## 16. Phase A: ShaSpec + Gating (150ep fullscale (MIMIC + ReXGradient))

### 1) Introduction
Long-run verification of Phase A on larger/clinically relevant datasets.

### 2) Methodology (implementation w.r.t. baseline)
Same Phase A model as above, extended to 150 epochs on full MIMIC and ReXGradient.
Source implementation/result file: `experiments/phaseA_shaspec_gating_longrun_mimic_rex/phaseA_shaspec_gating_longrun_mimic_rex.csv`

### 3) Experiments conducted
- Number of experiment runs: **2 dataset runs**
- Datasets covered in this run: mimic, rexgradient

### 4) Results
| Dataset | Technique AUC | Baseline AUC | Delta |
|---|---:|---:|---:|
| mimic | 0.7975 | 0.7382 | +0.0593 |
| rexgradient | 0.9261 | 0.8948 | +0.0313 |

### 5) Comparison with baseline (which is better and why)
- Long-run subset verdict: better on 2/2 datasets (mimic, rexgradient).
- Interpretation: extended epochs helped ReXGradient strongly, while MIMIC saturated relative to the 50-epoch Phase A result.

## Final Conclusion

- The strongest broad result in this workspace is **Phase A: ShaSpec + Gating (50ep)** across all 4 datasets (highest mean AUC among full 4-dataset comparisons).
- **Phase A long-run (150ep)** confirms additional gain on ReXGradient, with MIMIC plateauing.
- Techniques that consistently underperform baseline (e.g., uncertainty-aware fusion, set/perceiver in these runs) should be positioned as negative/ablation findings in thesis writing.