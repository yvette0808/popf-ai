# POPF ResNet-18 Frozen Baseline vs Layer4 Fine-tuning

Both experiments are `2026 Reconstruction` experiments. Neither result is a 2019 Original result.

## Controlled conditions

- Same modern_v1 dataset, label mapping, and fixed `154/34/34` split.
- Same seed `42`, batch size `16`, epochs `20`, weight decay `1e-4`, AdamW, RGB `224x224`, ImageNet normalization, and no augmentation.
- Same train-only class weights.
- Test was not used for model selection and was evaluated once per experiment after validation selection.
- The only intended model variable is trainability: baseline trains `fc`; fine-tuning trains `layer4 + fc` with separate learning rates.

| Metric | Frozen baseline | Layer4 fine-tuning | Difference |
|---|---:|---:|---:|
| Validation Accuracy | 0.676471 | 0.794118 | +0.117647 |
| Validation Macro-F1 | 0.671457 | 0.797959 | +0.126502 |
| Test Accuracy | 0.441176 | 0.617647 | +0.176471 |
| Test Macro-F1 | 0.481793 | 0.554313 | +0.072520 |
| Test Macro Precision | 0.578571 | 0.608844 | +0.030272 |
| Test Macro Recall | 0.466357 | 0.543476 | +0.077118 |
| Test Top-3 Accuracy | 0.852941 | 0.882353 | +0.029412 |

## Validation per-class F1 at selected checkpoints

- Baseline selected epoch: `13`
- Fine-tuning selected epoch: `4`

| Label | Frozen baseline | Layer4 fine-tuning | Difference |
|---|---:|---:|---:|
| 三块瓦脸 | 0.588235 | 0.833333 | +0.245098 |
| 碎脸 | 0.923077 | 0.666667 | -0.256410 |
| 象形脸 | 0.833333 | 1.000000 | +0.166667 |
| 花三块瓦脸 | 0.222222 | 0.285714 | +0.063492 |
| 整脸 | 0.666667 | 1.000000 | +0.333333 |
| 十字门脸 | 0.800000 | 0.800000 | +0.000000 |
| 六分脸 | 0.666667 | 1.000000 | +0.333333 |

## Test per-class F1

| Label | Frozen baseline | Layer4 fine-tuning | Difference |
|---|---:|---:|---:|
| 三块瓦脸 | 0.705882 | 0.869565 | +0.163683 |
| 碎脸 | 0.266667 | 0.428571 | +0.161905 |
| 象形脸 | 0.400000 | 0.615385 | +0.215385 |
| 花三块瓦脸 | 0.000000 | 0.000000 | +0.000000 |
| 整脸 | 0.500000 | 0.800000 | +0.300000 |
| 十字门脸 | 0.500000 | 0.500000 | +0.000000 |
| 六分脸 | 1.000000 | 0.666667 | -0.333333 |

## Confusion matrices

Baseline test confusion matrix:

```text
[[6, 1, 2, 1, 0, 1, 0], [0, 2, 3, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 3, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 2]]
```

Layer4 fine-tuning test confusion matrix:

```text
[[10, 1, 0, 0, 0, 0, 0], [0, 3, 1, 2, 0, 1, 0], [0, 1, 4, 0, 0, 0, 0], [1, 2, 1, 0, 0, 0, 0], [1, 0, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1]]
```

## Prediction-level comparison

- `artifacts/experiments/resnet18_layer4_finetune/baseline_vs_finetune_test_predictions.csv` records both models' Top-3 rankings for each test image.
- `reports/resnet18_layer4_finetune_test_errors.png` shows fine-tuning top-1 errors for manual inspection.

No next experiment was executed automatically.
