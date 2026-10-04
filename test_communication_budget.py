import unittest

import numpy as np

from communication_budget import taichi_budget, tokcom_budget
from digital_link import transmit_tokens


class CommunicationBudgetTests(unittest.TestCase):
    def test_reference_budget_matches_noiseless_link_instrumentation(self):
        expected = tokcom_budget(
            num_tokens=256,
            codebook_size=1024,
            image_height=256,
            image_width=256,
            packet_tokens=4,
        )
        _, missing, measured = transmit_tokens(
            np.zeros((1, 256), dtype=np.int64),
            codebook_size=1024,
            snr_db=100.0,
            channel="noiseless",
            seed=1,
            packet_tokens=4,
        )
        self.assertFalse(missing.any())
        self.assertEqual(expected["source_bits"], measured["payload_bits"])
        self.assertEqual(expected["coded_bits"], measured["coded_bits"])
        self.assertEqual(expected["complex_channel_symbols"], measured["symbols"])
        self.assertEqual(expected["qam_padding_bits"], measured["qam_padding_bits"])
        self.assertEqual(expected["packets"], measured["packets"])

    def test_partial_last_packet_is_counted_exactly(self):
        budget = tokcom_budget(
            num_tokens=5,
            codebook_size=8,
            image_height=1,
            image_width=5,
            packet_tokens=4,
        )
        # Payloads are 12 and 3 bits. Each packet adds 16 CRC and 6 tail bits,
        # is encoded at rate 1/2, and is padded independently to 16-QAM groups.
        self.assertEqual(budget["packets"], 2)
        self.assertEqual(budget["source_bits"], 15)
        self.assertEqual(budget["coded_bits"], 118)
        self.assertEqual(budget["qam_padding_bits"], 2)
        self.assertEqual(budget["complex_channel_symbols"], 30)

    def test_stage11_semantic_channel_uses_iq_pairs(self):
        budget = taichi_budget(
            visual_tokens=16,
            real_channel_dimension=64,
            image_height=224,
            image_width=224,
        )
        self.assertEqual(budget["real_channel_values"], 1024)
        self.assertEqual(budget["complex_channel_symbols"], 512)
        self.assertIsNone(budget["bit_rate"])

    def test_semantic_channel_width_must_be_even(self):
        with self.assertRaises(ValueError):
            taichi_budget(
                visual_tokens=16,
                real_channel_dimension=63,
                image_height=224,
                image_width=224,
            )


if __name__ == "__main__":
    unittest.main()
