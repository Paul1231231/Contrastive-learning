"""Data augmentation helpers for contrastive learning."""

from typing import Callable, Tuple


def make_two_view_transform(base_transform: Callable) -> Callable:
    """Return a transform that produces two augmented views of one sample."""

    def _transform(x) -> Tuple:
        return base_transform(x), base_transform(x)

    return _transform
