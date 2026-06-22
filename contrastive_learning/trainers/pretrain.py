"""Pretraining loop skeleton for contrastive learning."""

from contrastive_learning.losses.nt_xent import nt_xent_loss


def train_one_epoch(model, projection_head, dataloader, optimizer):
    """Framework-agnostic starter loop.

    Integrate device transfer, backward, and optimizer steps for your framework.
    """
    losses = []
    for view1, view2, _ in dataloader:
        z1 = projection_head(model(view1))
        z2 = projection_head(model(view2))
        losses.append(nt_xent_loss(z1, z2))
        optimizer.zero_grad()
        optimizer.step()
    return losses
