import copy,hashlib,unittest,tempfile
from pathlib import Path
from imagenet100_manifest import validate,verify_files
class Contracts(unittest.TestCase):
    def setUp(self):
        self.classes=[f'n{i:08d}' for i in range(100)]
        self.m=dict(classes=self.classes,class_list_source='synthetic unit fixture',split_source='synthetic unit fixture',
            images=[dict(path=f'{s}/{c}.jpg',split=s,synset=c,sha256=hashlib.sha256((s+c).encode()).hexdigest()) for s in ('train','test') for c in self.classes])
    def test_structure_not_provenance(self):
        r=validate(self.m,set(self.classes));self.assertTrue(r['structural_checks_passed']);self.assertFalse(r['paper_dataset_identity_established'])
    def test_99_classes(self):
        m=copy.deepcopy(self.m);m['classes']=m['classes'][:-1]
        with self.assertRaises(ValueError):validate(m,set(self.classes))
    def test_duplicate_class(self):
        m=copy.deepcopy(self.m);m['classes'][1]=m['classes'][0]
        with self.assertRaises(ValueError):validate(m,set(self.classes))
    def test_missing_test_class(self):
        m=copy.deepcopy(self.m);m['images']=m['images'][:-1]
        with self.assertRaises(ValueError):validate(m,set(self.classes))
    def test_cross_split_content(self):
        m=copy.deepcopy(self.m);m['images'][100]['sha256']=m['images'][0]['sha256']
        with self.assertRaises(ValueError):validate(m,set(self.classes))
    def test_traversal(self):
        m=copy.deepcopy(self.m);m['images'][0]['path']='../outside.jpg'
        with self.assertRaises(ValueError):validate(m,set(self.classes))
    def test_frozen_holdout_in_training(self):
        with self.assertRaises(ValueError):validate(self.m,set(self.classes),[self.m['images'][0]['sha256']])
    def test_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'a.jpg';p.write_bytes(b'identity fixture')
            m={'images':[dict(path='a.jpg',sha256=hashlib.sha256(p.read_bytes()).hexdigest())]}
            self.assertEqual(verify_files(m,directory),1);p.write_bytes(b'changed fixture')
            with self.assertRaises(ValueError):verify_files(m,directory)
    def test_file_root_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'dataset';root.mkdir();p=Path(directory)/'outside.jpg';p.write_bytes(b'fixture')
            m={'images':[dict(path='../outside.jpg',sha256=hashlib.sha256(p.read_bytes()).hexdigest())]}
            with self.assertRaises(ValueError):verify_files(m,root)
def verify():
    r=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Contracts).run(r)
    assert r.wasSuccessful(),(r.errors,r.failures)
    return dict(passed=r.testsRun,failed=len(r.errors)+len(r.failures))
if __name__=='__main__':print(verify())
