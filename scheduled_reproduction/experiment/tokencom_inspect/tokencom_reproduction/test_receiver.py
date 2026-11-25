import unittest
import numpy as np
from run_pretrained import spatial_fill


class ReceiverTests(unittest.TestCase):
    def test_only_receiver_context_used(self):
        mask=np.array([[False,True,True,False]])
        first=np.array([[5,123,456,9]])
        second=np.array([[5,888,999,9]])
        # Erased values are unknown: changing them cannot affect the result.
        np.testing.assert_array_equal(spatial_fill(first,mask),spatial_fill(second,mask))
        self.assertEqual(spatial_fill(first,mask)[0,0],5)
        self.assertEqual(spatial_fill(first,mask)[0,3],9)

    def test_total_erasure_fallback(self):
        np.testing.assert_array_equal(spatial_fill(np.full((1,4),77),np.ones((1,4),bool)),np.zeros((1,4)))


if __name__=='__main__': unittest.main(verbosity=2)
