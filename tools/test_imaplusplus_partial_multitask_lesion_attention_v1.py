from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
    IdentityInitializedECA,
)
from imaplusplus_partial_multitask_v1 import BinaryPartialMultiTaskModel, ContractError


class Run12AttentionTests(unittest.TestCase):
    def test_eca_is_exact_identity_at_initialization(self) -> None:
        module = IdentityInitializedECA(kernel_size=5)
        feature = torch.randn(2, 16, 7, 9)
        self.assertTrue(torch.equal(module(feature), feature))

    def test_eca_rejects_even_kernel(self) -> None:
        with self.assertRaisesRegex(ContractError, "positive odd"):
            IdentityInitializedECA(kernel_size=4)

    def test_binary_foreground_is_sigmoid_and_detached(self) -> None:
        head = BinaryLesionGuidedAttentionHead(nc=1, ch=(8, 16, 32), hidden_channels=8)
        logits = torch.tensor([[[[-2.0, 0.0, 2.0]]]], requires_grad=True)
        foreground = head._foreground_probability(logits)
        self.assertTrue(torch.allclose(foreground, logits.detach().sigmoid()))
        self.assertFalse(foreground.requires_grad)

    def test_guided_pooling_uses_lesion_region(self) -> None:
        head = BinaryLesionGuidedAttentionHead(
            nc=1,
            ch=(1, 1, 1),
            hidden_channels=1,
            mask_guided_mix=0.5,
        )
        feature = torch.tensor([[[[1.0, 1.0], [1.0, 9.0]]]])
        foreground = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
        pooled = head._pool_scale(feature, foreground, 0)
        # global mean=3 and lesion-guided mean=9, fixed 50/50 mixture=6.
        self.assertTrue(torch.allclose(pooled, torch.tensor([[6.0]])))

    def test_training_forward_shapes_and_classifier_gradient_scope(self) -> None:
        head = BinaryLesionGuidedAttentionHead(nc=1, ch=(8, 16, 32), hidden_channels=8)
        head.train()
        features = [
            torch.randn(2, 8, 16, 16, requires_grad=True),
            torch.randn(2, 16, 8, 8, requires_grad=True),
            torch.randn(2, 32, 4, 4, requires_grad=True),
        ]
        semantic, auxiliary, image_logits = head(features)
        self.assertEqual(tuple(semantic.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(auxiliary.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(image_logits.shape), (2, 8))
        image_logits.sum().backward()
        self.assertIsNone(head.classifier[0].conv.weight.grad)
        self.assertIsNotNone(head.image_classifier[0].weight.grad)
        for feature in features:
            self.assertIsNotNone(feature.grad)

    def test_head_requires_three_scales(self) -> None:
        head = BinaryLesionGuidedAttentionHead(nc=1, ch=(8, 16, 32), hidden_channels=8)
        with self.assertRaisesRegex(ContractError, "P3, P4, P5"):
            head([torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)])

    def test_full_model_preserves_run10_initialization_and_adds_zero_p5_block(self) -> None:
        torch.manual_seed(42)
        baseline = BinaryPartialMultiTaskModel(verbose=False)
        torch.manual_seed(42)
        attention = BinaryLesionGuidedAttentionModel(verbose=False)
        baseline_head = baseline.model[-1]
        attention_head = attention.model[-1]
        self.assertEqual(list(attention_head.f), [16, 13, 10])
        self.assertIn(10, attention.save)
        self.assertEqual(tuple(attention_head.scale_channels), (128, 256, 512))
        for left, right in zip(
            baseline_head.classifier.state_dict().values(),
            attention_head.classifier.state_dict().values(),
        ):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(
            baseline_head.aux_head.state_dict().values(),
            attention_head.aux_head.state_dict().values(),
        ):
            self.assertTrue(torch.equal(left, right))
        old_first = baseline_head.image_classifier[0]
        new_first = attention_head.image_classifier[0]
        self.assertTrue(torch.equal(old_first.weight, new_first.weight[:, : old_first.in_features]))
        self.assertTrue(bool((new_first.weight[:, old_first.in_features :] == 0).all().item()))
        self.assertTrue(torch.equal(old_first.bias, new_first.bias))
        self.assertTrue(
            torch.equal(baseline_head.image_classifier[3].weight, attention_head.image_classifier[3].weight)
        )


if __name__ == "__main__":
    unittest.main()
