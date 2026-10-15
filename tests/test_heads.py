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

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"),
                         "Gemma integration dependencies are not installed")
    def test_actual_gemma4_shared_kv_and_lora_gradients(self):
        import torch
        from torch.nn import functional as F
        from transformers import Gemma4TextConfig, Gemma4TextModel
        from peft import LoraConfig, get_peft_model
        from gemma_tabular.heads import NumericalModel, digit_loss

        config = Gemma4TextConfig(
            vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=6, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=4, num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention"] * 3,
            sliding_window=16, max_position_embeddings=64,
        )
        ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
        mask = (ids != 0).long()
        for kind in ("digits", "regression"):
            with self.subTest(head=kind):
                backbone = Gemma4TextModel(config)
                for parameter in backbone.parameters():
                    parameter.requires_grad = False
                backbone = get_peft_model(backbone, LoraConfig(
                    r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model = NumericalModel(backbone, 16, kind)
                scores = model(ids, mask)
                loss = (digit_loss(scores, torch.zeros(2, 8, dtype=torch.long)) if kind == "digits"
                        else F.mse_loss(scores, torch.tensor([0., 1.])))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
                gradients = [parameter.grad for name, parameter in backbone.named_parameters()
                             if "lora_" in name and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
