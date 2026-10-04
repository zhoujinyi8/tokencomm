import unittest
import torch

from train_qwen_dual_taichi import balanced_indices


class Stage9ProtocolTests(unittest.TestCase):
    def test_reduced_validation_subset_is_ten_class_balanced(self):
        labels=torch.arange(10).repeat_interleave(15)
        selected=balanced_indices(labels,64)
        counts=torch.bincount(labels[selected],minlength=10)
        self.assertEqual(len(selected),64)
        self.assertEqual(int(counts.min()),6)
        self.assertEqual(int(counts.max()),7)
        self.assertEqual(int((counts>0).sum()),10)


if __name__=='__main__': unittest.main()
