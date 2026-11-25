import unittest
import numpy as np
from digital_link import (crc16,append_crc,check_crc,conv_encode,viterbi,qam16,qam_llr,
                          CONSTELLATION,BIT_LABELS,transmit_tokens,transmit_packet)


class DigitalLinkTests(unittest.TestCase):
    def test_crc_known_vector_and_single_bit_errors(self):
        bits=np.unpackbits(np.frombuffer(b'123456789',dtype=np.uint8))
        self.assertEqual(crc16(bits),0x29b1)
        encoded=append_crc(bits); self.assertTrue(check_crc(encoded))
        for k in range(len(encoded)):
            corrupt=encoded.copy(); corrupt[k]^=1
            self.assertFalse(check_crc(corrupt))

    def test_code_roundtrip_with_known_impulse_response(self):
        # (171,133) octal, least-significant/newest bit first.
        self.assertEqual(conv_encode(np.array([1],dtype=np.uint8)).tolist(),
                         [1,1,0,1,0,0,1,1,1,1,1,0,1,1])
        for n in [1,24,40,80]:
            bits=np.random.default_rng(n).integers(0,2,n,dtype=np.uint8)
            code=conv_encode(bits); llr=(1-2*code.astype(float))*8
            np.testing.assert_array_equal(viterbi(llr),bits)

    def test_qam_energy_gray_and_soft_sign(self):
        self.assertAlmostEqual(float(np.abs(CONSTELLATION).mean()**2),.8972135955,places=6)
        self.assertAlmostEqual(float((np.abs(CONSTELLATION)**2).mean()),1.)
        symbols,padding=qam16(BIT_LABELS.flatten())
        self.assertEqual(padding,0)
        np.testing.assert_array_equal(qam_llr(symbols,np.ones(16),.1)<0,BIT_LABELS.flatten())
        for i,z in enumerate(CONSTELLATION):
            dist=np.abs(CONSTELLATION-z); near=np.where(np.isclose(dist,2/np.sqrt(10)))[0]
            for j in near: self.assertEqual(int((BIT_LABELS[i]!=BIT_LABELS[j]).sum()),1)

    def test_full_link_no_noise_and_short_last_packet(self):
        ids=np.random.default_rng(1).integers(0,1024,(3,17))
        received,missing,stats=transmit_tokens(ids,channel='noiseless')
        np.testing.assert_array_equal(received,ids)
        self.assertFalse(missing.any()); self.assertEqual(stats['packets'],15)
        self.assertEqual(stats['payload_bits'],510)

    def test_high_snr_and_low_snr(self):
        ids=np.random.default_rng(1).integers(0,1024,(6,64))
        high,hm,hs=transmit_tokens(ids,snr_db=30)
        _,lm,ls=transmit_tokens(ids,snr_db=-5)
        np.testing.assert_array_equal(high,ids); self.assertFalse(hm.any())
        self.assertGreater(float(lm.mean()),.5)
        self.assertGreater(ls['payload_ber'],hs['payload_ber'])


if __name__=='__main__': unittest.main(verbosity=2)
