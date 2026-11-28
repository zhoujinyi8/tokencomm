import unittest
from cheng_packet_budget import packet_budget,packetize,reassemble
class Contracts(unittest.TestCase):
    def test_exact_multiple(self):self.assertEqual(packet_budget(320)['packets'],64)
    def test_tail(self):
        b=packet_budget(321);self.assertEqual((b['packets'],b['tail_padding_bits']),(65,32))
    def test_wire_units(self):
        b=packet_budget(320);self.assertEqual(b['forward_qam_symbols'],1664);self.assertEqual(b['forward_wire_bits'],b['convolutional_bits']+b['qam_padding_bits'])
    def test_roundtrip(self):
        data=bytes(range(256))*3+b'a';p,b=packetize(data);self.assertEqual(reassemble(p,len(data)),data);self.assertEqual(len(p),b['packets'])
    def test_nonzero_tail(self):
        p,b=packetize(bytes(321));p[-1]=p[-1][:-1]+b'x'
        with self.assertRaises(ValueError):reassemble(p,321)
    def test_invalid_size(self):
        with self.assertRaises(ValueError):packet_budget(12)
        with self.assertRaises(ValueError):reassemble([b'ab'],2)
def verify():
    result=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Contracts).run(result)
    assert result.wasSuccessful(),(result.errors,result.failures)
    return dict(passed=result.testsRun,failed=len(result.errors)+len(result.failures))
if __name__=='__main__':print(verify())
