# POPF Modern Dataset

This directory is a 2026 Reconstruction dataset preparation output. It is built from the audited 272-image historical set and does not modify `data/raw`.

## Files

- `modern_all_candidates.csv`: all 272 historical records, including raw metadata fields copied with `metadata_raw__` prefixes.
- `dataset_manifest.csv`: training-oriented manifest for all 272 records, with inclusion status and split.
- `label_inventory.csv`: raw `谱式` label counts and modern_v1 eligibility.
- `train.csv`, `validation.csv`, `test.csv`: modern_v1 split files.
- `dataset_config.json`: fixed seed, label policy, split policy, and provenance.
- `label_mapping.json`: fixed `raw_face_type` to integer ID mapping for all three splits.
- `dataset_report.md`: human-readable dataset analysis.

## Boundary

`mapping_status=user_confirmed_sequence_mapping` means image `001` is paired with metadata sequence `1`, image `002` with sequence `2`, and so on. This one-to-one relationship was confirmed by the project owner on 2026-09-20 for the current POPF dataset.

## modern_v1 policy

- Preserve raw labels exactly.
- Analyze labels using current XLS `谱式` values.
- Do not compress the observed labels into the paper's 16 classes.
- Exclude missing `谱式` values.
- Include non-empty classes with at least 10 samples.
- Use stratified split with seed `42`.
- Do not copy, augment, resize, clean, or rename raw images.

## Dataset and DataLoader

`src/datasets/popf_dataset.py` reads each `image_path` as
`archive.zip::member/name.jpg` and reads the ZIP member directly. It does not
extract the archive or create a second image directory.

The deterministic preprocessing pipeline is:

`JPEG -> RGB -> Resize -> float tensor -> ImageNet normalization`

The default input size is `224 x 224`, and the `image_size` argument can be
changed when building the loaders. No augmentation is applied.

Example:

```python
from src.datasets.popf_dataset import build_dataloaders

train_loader, validation_loader, test_loader = build_dataloaders(
    image_size=(224, 224),
    batch_size=32,
    num_workers=0,
    seed=42,
)
```

## Training note

These files prepare data for a later model phase only. No model has been trained by this stage, and no accuracy or other model metric is created here.
