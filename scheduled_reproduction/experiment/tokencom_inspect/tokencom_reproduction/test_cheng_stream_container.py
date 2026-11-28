import hashlib,json,struct,unittest
from pathlib import Path
from cheng_stream_container import HEADER,MODELS,MAX_BYTES,pack,unpack
class Contracts(unittest.TestCase):
    def setUp(self):self.data=pack(MODELS[0],1,256,256,(4,4),[b'abcdef',b'1234'])
    def test_models_roundtrip(self):
        for model in MODELS:
            v=unpack(pack(model,1,256,256,(4,4),[b'abcdef',b'1234']))
            self.assertEqual(v['model'],model);self.assertEqual(v['streams'],[b'abcdef',b'1234']);self.assertEqual(v['shape'],(4,4))
    def test_exact_header_bits(self):
        self.assertEqual(unpack(self.data)['header_bytes'],34);self.assertEqual(len(self.data)*8,34*8+10*8)
    def test_truncated_header(self):
        with self.assertRaises(ValueError):unpack(self.data[:20])
    def test_truncated_payload(self):
        with self.assertRaises(ValueError):unpack(self.data[:-1])
    def test_extra_bytes(self):
        with self.assertRaises(ValueError):unpack(self.data+b'x')
    def test_unknown_version(self):
        changed=bytearray(self.data);changed[4]=2
        with self.assertRaises(ValueError):unpack(bytes(changed))
    def test_invalid_geometry_quality(self):
        with self.assertRaises(ValueError):pack(MODELS[0],0,256,256,(4,4),[b'a',b'b'])
        with self.assertRaises(ValueError):pack(MODELS[0],1,0,256,(4,4),[b'a',b'b'])
    def test_oversize_length(self):
        changed=bytearray(self.data);struct.pack_into('>I',changed,HEADER.size,MAX_BYTES+1)
        with self.assertRaises(ValueError):unpack(bytes(changed))
def verify():
    result=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Contracts).run(result)
    assert result.wasSuccessful(),(result.errors,result.failures)
    return dict(passed=result.testsRun,failed=len(result.errors)+len(result.failures))
if __name__=='__main__':
    r=Path(__file__).parent;v=verify();v.update(helper_sha256=hashlib.sha256((r/'cheng_stream_container.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (r/'runs/stage65_container_contract.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(v)
