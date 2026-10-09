import math
import unittest

import torch

from tokcom_image_metrics import psnr_per_image


class TokComImageMetricTests(unittest.TestCase):
    def test_psnr_identity_and_known_error(self):
        original = torch.zeros(2, 3, 4, 4)
        reconstructed = original.clone()
        reconstructed[1].fill_(0.1)
        result = psnr_per_image(original, reconstructed)
        self.assertTrue(math.isinf(result[0].item()))
        self.assertAlmostEqual(result[1].item(), 20.0, places=5)

    def test_psnr_rejects_invalid_ranges_and_shapes(self):
        with self.assertRaises(ValueError):
            psnr_per_image(torch.zeros(1, 3, 4, 4), torch.zeros(1, 1, 4, 4))
        with self.assertRaises(ValueError):
            psnr_per_image(torch.zeros(1, 3, 4, 4), torch.full((1, 3, 4, 4), 1.1))


if __name__ == "__main__":
    unittest.main()
