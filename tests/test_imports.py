def test_scaffold_imports():
    import contrastive_learning.data.augmentations  # noqa: F401
    import contrastive_learning.data.dataset  # noqa: F401
    import contrastive_learning.losses.nt_xent  # noqa: F401
    import contrastive_learning.models.encoder  # noqa: F401
    import contrastive_learning.models.momentum_encoder  # noqa: F401
    import contrastive_learning.models.projection_head  # noqa: F401
    import contrastive_learning.trainers.pretrain  # noqa: F401
