# Contrastive-learning

Starter scaffold for a contrastive learning project (data augmentation, end-to-end encoder, momentum encoder, and training entrypoints).

## Suggested file structure

```text
.
├── configs/
│   └── default.yaml
├── contrastive_learning/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentations.py
│   │   └── dataset.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── nt_xent.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── momentum_encoder.py
│   │   └── projection_head.py
│   └── trainers/
│       ├── __init__.py
│       └── pretrain.py
├── scripts/
│   └── train.py
└── tests/
    └── test_imports.py
```

## Notes

- `data/augmentations.py`: augmentation pipelines for contrastive views.
- `models/encoder.py`: base encoder backbone.
- `models/momentum_encoder.py`: online/target encoder momentum update logic.
- `losses/nt_xent.py`: contrastive objective (NT-Xent).
- `trainers/pretrain.py`: pretraining loop orchestration.
