# POPF ResNet-18 Three-Way Fine-tuning Schedule Comparison

All three entries are `2026 Reconstruction` experiments. None is a 2019 Original result.

## Controlled conditions

- Same `modern_v1` dataset, fixed `154/34/34` split, labels, input size, no augmentation, train-only class weights, seed `42`, batch size `16`, AdamW, and weight decay `1e-4`.
- The conservative run changes only the layer4/fc learning rates and adds validation macro-F1 early stopping.
- Test was not used for model selection; each test result comes from one final evaluation of the selected checkpoint.

| Metric | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |
|---|---:|---:|---:|
| Validation Accuracy | 0.676471 | 0.794118 | 0.764706 |
| Validation Macro-F1 | 0.671457 | 0.797959 | 0.633700 |
| Test Accuracy | 0.441176 | 0.617647 | 0.617647 |
| Test Macro-F1 | 0.481793 | 0.554313 | 0.591095 |
| Test Macro Precision | 0.578571 | 0.608844 | 0.701299 |
| Test Macro Recall | 0.466357 | 0.543476 | 0.543476 |
| Test Top-3 Accuracy | 0.852941 | 0.882353 | 0.882353 |

## Training and selection behavior

| Experiment | Epochs trained | Early stopped | Best epoch | Train accuracy at best | Accuracy gap at best | Val macro-F1 range | Val macro-F1 std |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18_baseline | 20 | False | 13 | 0.805195 | +0.128724 | 0.524271 | 0.132058 |
| resnet18_layer4_finetune | 20 | False | 4 | 0.987013 | +0.192895 | 0.228594 | 0.057911 |
| resnet18_conservative_finetune | 7 | True | 4 | 0.948052 | +0.183346 | 0.207378 | 0.062103 |

## Validation per-class F1 at selected checkpoints

| Label | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |
|---|---:|---:|---:|
| 三块瓦脸 | 0.588235 | 0.833333 | 0.916667 |
| 碎脸 | 0.923077 | 0.666667 | 0.769231 |
| 象形脸 | 0.833333 | 1.000000 | 0.888889 |
| 花三块瓦脸 | 0.222222 | 0.285714 | 0.444444 |
| 整脸 | 0.666667 | 1.000000 | 0.750000 |
| 十字门脸 | 0.800000 | 0.800000 | 0.000000 |
| 六分脸 | 0.666667 | 1.000000 | 0.666667 |

## Test per-class F1

| Label | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |
|---|---:|---:|---:|
| 三块瓦脸 | 0.705882 | 0.869565 | 0.909091 |
| 碎脸 | 0.266667 | 0.428571 | 0.428571 |
| 象形脸 | 0.400000 | 0.615385 | 0.666667 |
| 花三块瓦脸 | 0.000000 | 0.000000 | 0.000000 |
| 整脸 | 0.500000 | 0.800000 | 0.800000 |
| 十字门脸 | 0.500000 | 0.500000 | 0.666667 |
| 六分脸 | 1.000000 | 0.666667 | 0.666667 |

## Test confusion matrices

resnet18_baseline:

```text
[[6, 1, 2, 1, 0, 1, 0], [0, 2, 3, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 3, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 2]]
```

resnet18_layer4_finetune:

```text
[[10, 1, 0, 0, 0, 0, 0], [0, 3, 1, 2, 0, 1, 0], [0, 1, 4, 0, 0, 0, 0], [1, 2, 1, 0, 0, 0, 0], [1, 0, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1]]
```

resnet18_conservative_finetune:

```text
[[10, 1, 0, 0, 0, 0, 0], [0, 3, 0, 4, 0, 0, 0], [0, 1, 4, 0, 0, 0, 0], [1, 2, 1, 0, 0, 0, 0], [0, 0, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1]]
```

## Interpretation

The per-class and confusion-matrix changes should be read together with the very small support of `十字门脸` and `六分脸`, and the persistent difficulty of `花三块瓦脸`.
- A higher validation score alone is not treated as proof of improved generalization.
- The conservative schedule was selected only by validation macro-F1; its test metrics are reported after selection and were not used to change the schedule.

No next experiment was executed automatically.
