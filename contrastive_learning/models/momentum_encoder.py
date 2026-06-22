"""Momentum encoder wrapper (online + target encoders)."""

import copy


class MomentumEncoder:
    def __init__(self, online_encoder, momentum: float = 0.99):
        self.online_encoder = online_encoder
        self.target_encoder = copy.deepcopy(online_encoder)
        self.momentum = momentum

    def update_target(self):
        """Update target encoder parameters from online encoder.

        Implement parameter-wise EMA update with your framework tensors.
        """
