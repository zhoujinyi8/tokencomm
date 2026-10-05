import json
import unittest
from pathlib import Path


class Stage5Artifacts(unittest.TestCase):
    def test_new_split_is_disjoint_and_test_locked(self):
        root=Path('data/imagenette_stage5')
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['old_pilot_overlap'],0)
        self.assertEqual(manifest['cross_split_file_duplicates'],0)
        self.assertEqual(manifest['cross_split_decoded_pixel_duplicates'],0)
        self.assertEqual(manifest['split_counts'],{'train':800,'val':100,'test':100})
        summary=json.loads((root/'token_cache/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['splits'],['train','val']);self.assertFalse(summary['test_opened'])
        self.assertFalse((root/'token_cache/test.pt').exists())
        self.assertFalse((root/'resnet18_features/test.pt').exists())

    def test_training_runs_completed_without_test(self):
        for name in ['stage5_context_maskgit','stage5_pretrained_taichi']:
            status=json.loads((Path('runs')/name/'status.json').read_text(encoding='utf-8'))
            self.assertTrue(status['completed']);self.assertFalse(status['test_accessed'])


if __name__=='__main__':unittest.main()
