"""Projection head used for contrastive loss space."""


class ProjectionHead:
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def __call__(self, x):
        return x
