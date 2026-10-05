import io
import unittest

from PIL import Image

from prepare_stage12_holdout import reservoir_select


def image_bytes(rgb):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), rgb).save(stream, format="PNG")
    return stream.getvalue()


class Stage12HoldoutTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_excludes_file_hash(self):
        rows = []
        for label in range(2):
            for index in range(6):
                rows.append(
                    {
                        "label": label,
                        "image": {"bytes": image_bytes((label * 100, index * 20, 7)), "path": f"{label}-{index}.png"},
                    }
                )
        import hashlib

        excluded = hashlib.sha256(rows[0]["image"]["bytes"]).hexdigest()
        first, eligible, count = reservoir_select(rows, 2, 3, 123, {excluded}, set())
        second, eligible2, count2 = reservoir_select(rows, 2, 3, 123, {excluded}, set())
        self.assertEqual([[x["sha256"] for x in g] for g in first], [[x["sha256"] for x in g] for g in second])
        self.assertEqual(eligible, [5, 6])
        self.assertEqual((count, count2), (1, 1))
        self.assertEqual([len(group) for group in first], [3, 3])

    def test_pixel_duplicate_is_excluded(self):
        raw = image_bytes((1, 2, 3))
        from prepare_stage12_holdout import pixel_digest

        rows = [
            {"label": 0, "image": {"bytes": raw, "path": "duplicate.png"}},
            {"label": 0, "image": {"bytes": image_bytes((2, 3, 4)), "path": "keep.png"}},
        ]
        groups, eligible, excluded = reservoir_select(rows, 1, 1, 9, set(), {pixel_digest(raw)})
        self.assertEqual(eligible, [1])
        self.assertEqual(excluded, 1)
        self.assertEqual(groups[0][0]["source_path"], "keep.png")

    def test_insufficient_unseen_examples_fails_closed(self):
        rows = [{"label": 0, "image": {"bytes": image_bytes((1, 1, 1)), "path": "one.png"}}]
        with self.assertRaisesRegex(ValueError, "Insufficient unseen examples"):
            reservoir_select(rows, 1, 2, 1, set(), set())


if __name__ == "__main__":
    unittest.main()
