"""Two task heads reading exactly one pooled Gemma representation."""

import torch
from torch import nn
from torch.nn import functional as F

from .codec import SLOT_SIZES


def last_valid(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    indices = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
    if (indices < 0).any():
        raise ValueError("Cannot pool an empty input")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices].float()


class NumericalModel(nn.Module):
    def __init__(self, backbone, hidden_size, kind="digits"):
        super().__init__()
        if kind not in ("digits", "regression"):
            raise ValueError("Unsupported output head")
        self.backbone = backbone
        self.kind = kind
        self.head = nn.Linear(hidden_size, sum(SLOT_SIZES) if kind == "digits" else 1)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        representation = last_valid(output.last_hidden_state, attention_mask)
        scores = self.head(representation)
        return scores.split(SLOT_SIZES, dim=-1) if self.kind == "digits" else scores.squeeze(-1)


def digit_loss(logits, targets, prices=None, numerical_weight=0.0):
    if len(logits) != len(SLOT_SIZES):
        raise ValueError("Digit logits do not match the dollar-digit slots")
    categorical = torch.stack([F.cross_entropy(scores, targets[:, index]) for index, scores in enumerate(logits)]).mean()
    if numerical_weight == 0:
        return categorical
    if prices is None or (prices < 0).any():
        raise ValueError("The auxiliary numerical loss is defined for non-negative prices")
    probabilities = [scores.float().softmax(-1) for scores in logits]
    values = torch.arange(10, device=logits[0].device, dtype=torch.float32)
    expected = torch.zeros(logits[0].shape[0], device=logits[0].device, dtype=torch.float32)
    place = 1000
    for scores in probabilities:
        expected = expected + place * (scores * values).sum(-1)
        place //= 10
    numerical = F.smooth_l1_loss(torch.log1p(expected), torch.log1p(prices.float()))
    return categorical + numerical_weight * numerical
