# Thesis-Ready Experimental Report: Missing-Modality Strategies

## Experimental Setup

- Datasets: VQA-RAD, PathVQA, MIMIC, ReXGradient
- Evaluation metric: macro mean AUC (full condition) and worst-case missingness mean AUC (min across full/miss_img/miss_txt/miss_struct)
- Compared methods: Baseline (IRENE 10ep), Gating (10ep), Gating+Projection (10ep), Gating (50ep full-scale)

## Table 1. Dataset-level results and deltas vs baseline

| Dataset | Method | Full AUC | Δ Full vs Baseline | Worst-case AUC | Δ Worst vs Baseline |
|---|---:|---:|---:|---:|---:|
| vqarad | Baseline (IRENE, 10ep) | 0.9836 | +0.0000 | 0.5766 | +0.0000 |
| vqarad | Gating (10ep) | 0.9196 | -0.0640 | 0.9049 | +0.3283 |
| vqarad | Gating+Projection (10ep) | 0.9525 | -0.0311 | 0.9525 | +0.3759 |
| vqarad | Gating (50ep) | 0.9454 | -0.0382 | 0.9392 | +0.3626 |
| pathvqa | Baseline (IRENE, 10ep) | 0.8184 | +0.0000 | 0.5043 | +0.0000 |
| pathvqa | Gating (10ep) | 0.7425 | -0.0759 | 0.7406 | +0.2364 |
| pathvqa | Gating+Projection (10ep) | 0.7411 | -0.0773 | 0.7378 | +0.2335 |
| pathvqa | Gating (50ep) | 0.7955 | -0.0230 | 0.7582 | +0.2539 |
| mimic | Baseline (IRENE, 10ep) | 0.7298 | +0.0000 | 0.4240 | +0.0000 |
| mimic | Gating (10ep) | 0.7754 | +0.0456 | 0.7670 | +0.3431 |
| mimic | Gating+Projection (10ep) | 0.5858 | -0.1441 | 0.5858 | +0.1618 |
| mimic | Gating (50ep) | 0.7643 | +0.0345 | 0.7405 | +0.3165 |
| rexgradient | Baseline (IRENE, 10ep) | 0.8849 | +0.0000 | 0.6115 | +0.0000 |
| rexgradient | Gating (10ep) | 0.8490 | -0.0359 | 0.8486 | +0.2371 |
| rexgradient | Gating+Projection (10ep) | 0.8497 | -0.0352 | 0.8497 | +0.2382 |
| rexgradient | Gating (50ep) | 0.8770 | -0.0079 | 0.8770 | +0.2655 |

## Table 2. Average deltas across datasets (vs baseline)

| Method | Avg Δ Full AUC | Avg Δ Worst-case AUC |
|---|---:|---:|
| Gating (10ep) | -0.0325 | +0.2862 |
| Gating+Projection (10ep) | -0.0719 | +0.2523 |
| Gating (50ep) | -0.0086 | +0.2996 |

## Table 3.1. Disease-wise AUC comparison on VQARAD

| Disease | Baseline | Gating10 | Δ Gating10 | Gating+Proj10 | Δ Gating+Proj10 | Gating50 | Δ Gating50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COPD | NA | NA | NA | NA | NA | NA | NA |
| Bronchiectasis | NA | NA | NA | NA | NA | NA | NA |
| Pneumothorax | 0.9628 | 0.8139 | -0.1489 | 0.8765 | -0.0863 | 0.8697 | -0.0931 |
| Pneumonia | 0.9880 | 0.9450 | -0.0430 | 0.9811 | -0.0069 | 0.9665 | -0.0215 |
| ILD | NA | NA | NA | NA | NA | NA | NA |
| Tuberculosis | NA | NA | NA | NA | NA | NA | NA |
| Lung cancer | NA | NA | NA | NA | NA | NA | NA |
| Pleural effusion | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Table 3.2. Disease-wise AUC comparison on PATHVQA

| Disease | Baseline | Gating10 | Δ Gating10 | Gating+Proj10 | Δ Gating+Proj10 | Gating50 | Δ Gating50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COPD | NA | NA | NA | NA | NA | NA | NA |
| Bronchiectasis | NA | NA | NA | NA | NA | NA | NA |
| Pneumothorax | NA | NA | NA | NA | NA | NA | NA |
| Pneumonia | 0.8650 | 0.8601 | -0.0048 | 0.8851 | 0.0202 | 0.7276 | -0.1374 |
| ILD | 0.8588 | 0.6867 | -0.1721 | 0.6857 | -0.1731 | 0.6518 | -0.2071 |
| Tuberculosis | 0.5886 | 0.4880 | -0.1007 | 0.4210 | -0.1676 | 0.9610 | 0.3724 |
| Lung cancer | 0.9613 | 0.9353 | -0.0260 | 0.9726 | 0.0113 | 0.8416 | -0.1198 |
| Pleural effusion | NA | NA | NA | NA | NA | NA | NA |

## Table 3.3. Disease-wise AUC comparison on MIMIC

| Disease | Baseline | Gating10 | Δ Gating10 | Gating+Proj10 | Δ Gating+Proj10 | Gating50 | Δ Gating50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COPD | 0.5370 | 0.6404 | 0.1034 | 0.4653 | -0.0717 | 0.7582 | 0.2212 |
| Bronchiectasis | 0.6263 | 0.5574 | -0.0688 | 0.4568 | -0.1695 | 0.6112 | -0.0151 |
| Pneumothorax | 0.9071 | 0.8676 | -0.0395 | 0.8360 | -0.0710 | 0.8402 | -0.0668 |
| Pneumonia | 0.8645 | 0.8327 | -0.0318 | 0.6756 | -0.1888 | 0.8084 | -0.0560 |
| ILD | 0.9141 | 0.9055 | -0.0085 | 0.8951 | -0.0190 | 0.8736 | -0.0405 |
| Tuberculosis | 0.3622 | 0.8044 | 0.4422 | 0.3672 | 0.0050 | 0.6508 | 0.2886 |
| Lung cancer | 0.7105 | 0.6952 | -0.0154 | 0.3224 | -0.3882 | 0.6840 | -0.0265 |
| Pleural effusion | 0.9171 | 0.9004 | -0.0168 | 0.6677 | -0.2495 | 0.8883 | -0.0288 |

## Table 3.4. Disease-wise AUC comparison on REXGRADIENT

| Disease | Baseline | Gating10 | Δ Gating10 | Gating+Proj10 | Δ Gating+Proj10 | Gating50 | Δ Gating50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COPD | 0.8526 | 0.8424 | -0.0102 | 0.9437 | 0.0912 | 0.8508 | -0.0018 |
| Bronchiectasis | 0.9277 | 0.8001 | -0.1276 | 0.6149 | -0.3128 | 0.9247 | -0.0030 |
| Pneumothorax | 0.9783 | 0.9654 | -0.0129 | 0.9714 | -0.0068 | 0.9622 | -0.0161 |
| Pneumonia | 0.9542 | 0.9325 | -0.0216 | 0.9576 | 0.0034 | 0.9023 | -0.0518 |
| ILD | 0.9699 | 0.9688 | -0.0011 | 0.9726 | 0.0028 | 0.9543 | -0.0155 |
| Tuberculosis | 0.6197 | 0.5911 | -0.0286 | 0.6011 | -0.0186 | 0.7456 | 0.1259 |
| Lung cancer | 0.8007 | 0.7282 | -0.0726 | 0.7580 | -0.0427 | 0.7281 | -0.0726 |
| Pleural effusion | 0.9762 | 0.9638 | -0.0125 | 0.9780 | 0.0018 | 0.9477 | -0.0285 |

## Key Findings

- Baseline remains strongest on average full-AUC across datasets.
- Gating (50ep) recovers much of the full-AUC gap and clearly improves worst-case missingness robustness on all datasets.
- Gating+Projection (10ep) is robust in worst-case on VQA-RAD/ReXGradient but underperforms on PathVQA and especially MIMIC in full-AUC.
- For thesis positioning: report a trade-off between best full-condition accuracy (baseline) and robustness under missingness (gating variants), with Gating(50ep) as the best overall compromise among tested implementations.