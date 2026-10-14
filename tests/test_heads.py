import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class HeadTests(unittest.TestCase):
    def test_one_backbone_call_and_trainable_heads(self):
        import torch
        from torch import nn
        from types import SimpleNamespace
        from gemma_tabular.heads import NumericalModel, digit_loss

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(20, 4)
                self.calls = 0

            def forward(self, input_ids, attention_mask, use_cache=False):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        backbone = Backbone()
        model = NumericalModel(backbone, 4, "digits")
        scores = model(torch.tensor([[1, 2, 0], [0, 3, 4]]), torch.tensor([[1, 1, 0], [0, 1, 1]]))
        self.assertEqual(backbone.calls, 1)
        self.assertEqual([s.shape[-1] for s in scores], [2, 10, 10, 10, 10, 2, 10, 10])
        loss = digit_loss(scores, torch.zeros((2, 8), dtype=torch.long), torch.tensor([1., 2.]), 0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_last_nonpadding_position_for_either_padding_side(self):
        import torch
        from gemma_tabular.heads import last_valid
        hidden = torch.tensor([[[1.], [2.], [0.]], [[0.], [3.], [4.]]])
        pooled = last_valid(hidden, torch.tensor([[1, 1, 0], [0, 1, 1]]))
        self.assertEqual(pooled.tolist(), [[2.], [4.]])
