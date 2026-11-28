"""Check focused numerical facts against the original attachment's saved journal-text extraction."""
import argparse,hashlib,json,re,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--journal-text',type=Path,default=Path('D:/新建文件夹 (2)/2026-09-21/ban/tmp/pdfs/Token_Communications_A_Large_Model-Driven_Framework_for_Cross-Modal_Context-Aware_Semantic_Communications.txt'));args=p.parse_args()
r=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
source=args.journal_text.read_bytes();journal=source.decode('utf-8');compact=re.sub(r'\s+','',journal)
assert 'IEEEWirelessCommunications' in compact and 'October2025' in compact
assert 'ImageNet100' in compact and '16-QAM' in compact and '0.05MHz' in compact
assert '7-bit' in compact and '0.039bitperpixel' in compact
assert 'N=256' in compact and 'Q=1024' in compact and '4tokens' in compact
start=compact.index('“Cheng[11]”');fragment=compact[start:start+350]
assert 'fine-tuned' in fragment and '0.04' in fragment
needle='TERvaluesbeforeandaftertransformer-basedcontextprocessingare'
start=compact.index(needle);ter_fragment=compact[start:start+500]
arrays=re.findall(r'\[([0-9.,]+)\]',ter_fragment)[:3]
numeric=[[float(x) for x in a.split(',')] for a in arrays]
expected=[[.15,.1,.065,.038,.02,.005],[.03,.025,.019,.0126,.00764,.0021],[6.,6.5,7.,7.5,8.,9.]]
assert numeric==expected
fetch=json.loads((r/'runs/stage66_paper_source_fetch.json').read_text(encoding='utf-8'));arxiv=[]
for item in fetch['sources']:
    path=r/item['archive_path'];assert sha(path.read_bytes())==item['archive_sha256']
    with tarfile.open(path,'r:*') as archive:text=archive.extractfile('main_arxiv.tex').read().decode('utf-8')
    normalized=re.sub(r'\s+','',text);start=normalized.index('receivedTERvaluesbeforeandafter')
    arrays=re.findall(r'\[([0-9.,]+)\]',normalized[start:start+650])[:3]
    values=[[float(x) for x in a.split(',')] for a in arrays];assert values==expected;arxiv.append(dict(version=item['version'],ter_snr_arrays_equal=True))
receipt=dict(all_checks_passed=True,script_sha256=sha(Path(__file__).read_bytes()),
    journal_text_cache_filename=args.journal_text.name,journal_text_cache_sha256=sha(source),
    evidence_scope='User-provided journal text extraction cache; original PDF not re-rendered or re-hashed in this check',
    verified_journal_case_study_facts=dict(dataset='ImageNet100',tokens_per_image=256,codebook_size=1024,tokens_per_packet=4,
        payload_bpp=.039,cheng_finetuned_target_bpp=.04,modulation='16-QAM',bandwidth_mhz=.05,cmi_class_bits=7),
    ter_before=expected[0],ter_after=expected[1],snr_db=expected[2],arxiv_agreement=arxiv,
    focused_numerical_agreement_established=True,full_journal_source_equivalence_established=False,
    original_pdf_identity_reverified=False,author_code_numerical_equivalence_established=False,
    exact_100_class_manifest_established=False,cheng_recipe_or_checkpoint_identified=False)
(r/'runs/stage66_journal_extract_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
