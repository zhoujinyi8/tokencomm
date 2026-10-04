import json
import unittest
from pathlib import Path


class Stage7DataProtocol(unittest.TestCase):
    def test_third_split_is_disjoint_and_test_is_unopened(self):
        root = Path('data/imagenette_stage7')
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['split_counts'], {'train': 1000, 'val': 150, 'test': 150})
        self.assertEqual(manifest['per_class'], {'train': 100, 'val': 15, 'test': 15})
        self.assertEqual(manifest['cross_split_file_duplicates'], 0)
        self.assertEqual(manifest['cross_split_decoded_pixel_duplicates'], 0)
        self.assertGreater(manifest['excluded_previous']['train_hashes'], 0)
        self.assertGreater(manifest['excluded_previous']['holdout_hashes'], 0)
        self.assertEqual(manifest['test_policy'], 'Unused until v2 configuration is frozen.')

        # The VQ cache job is intentionally allowed to write only train/val.
        self.assertFalse((root / 'token_cache' / 'test.pt').exists())


if __name__ == '__main__':
    unittest.main()
