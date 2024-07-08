import torch
import torch.distributions as D
import torch.nn.functional as F

from enum import Enum


def ce_loss(logits, target, weights=None):
    return F.cross_entropy(logits, target, weight=weights, reduction='none')


def bce_loss(logits, target, weights=None):
    return F.binary_cross_entropy_with_logits(logits, target, reduction='none')


def focal_loss(logits, target, weights=None, n=2):
    c = logits.shape[1]
    x = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, c)
    target = target.argmax(dim=1).long()
    target = target.view(-1)

    log_p = F.log_softmax(x, dim=-1)
    ce = F.nll_loss(log_p, target, weight=weights, reduction='none')
    all_rows = torch.arange(len(x))
    log_pt = log_p[all_rows, target]

    pt = log_pt.exp()
    focal_term = (1 - pt + 1e-12) ** n

    loss = focal_term * ce
    return loss.view(-1, 200, 200)


def focal_loss_o(logits, target, weights=None, n=2):
    target = target.argmax(dim=1)
    log_p = F.log_softmax(logits, dim=1)

    ce = F.nll_loss(log_p, target, weight=weights, reduction='none')

    log_pt = log_p.gather(1, target[None])
    pt = log_pt.exp()
    loss = ce * (1 - pt + 1e-8) ** n

    return loss


def entropy_reg(alpha, beta_reg=.001):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg


def gamma(x):
    return torch.exp(torch.lgamma(x))
