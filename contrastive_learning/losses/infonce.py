"""NT-Xent loss wrapper."""
import torch
import torch.nn.functional as F


def approx_infoNCE_loss(q, k, temperature=0.7):
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    logits = torch.matmul(q, k.T) / temperature

    labels = torch.arange(q.size(0), device=q.device)

    loss = F.cross_entropy(logits, labels)
    return loss

