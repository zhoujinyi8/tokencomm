"""Compare primary arXiv sources without exporting copyrighted manuscript text."""
import hashlib,json,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent;run=r/'runs'
sha=lambda data:hashlib.sha256(data).hexdigest()
fetch=json.loads((run/'stage66_paper_source_fetch.json').read_text(encoding='utf-8'))
texts=[];members=[]
for item in fetch['sources']:
    assert item['status']=='downloaded_and_tar_verified'
    path=r/item['archive_path'];assert sha(path.read_bytes())==item['archive_sha256']
    with tarfile.open(path,'r:*') as archive:
        data=archive.extractfile('main_arxiv.tex').read()
        assert sha(data)==next(v['sha256'] for v in item['tex'] if v['name']=='main_arxiv.tex')
        texts.append(data.decode('utf-8'))
        files={}
        for entry in archive.getmembers():
            if entry.isfile():files[entry.name]=sha(archive.extractfile(entry).read())
        members.append(files)
expected=' The source code is publicly available at \\url{https://github.com/liqiao19/TokenCom_Code}.'
assert texts[1].count(expected)==1 and texts[1].replace(expected,'',1)==texts[0]
addition=expected
case_title=r'\section{Case Study: Cross-Modality TokCom for Generative Image Semantic Communication}'
next_title=r'\section{Open Problems and Future Research Directions}'
sections=[s[s.index(case_title):s.index(next_title)] for s in texts]
assert sections[0]==sections[1]
changed_members=sorted(n for n in set(members[0])|set(members[1]) if members[0].get(n)!=members[1].get(n))
assert changed_members==['00README.json','main_arxiv.tex']
metadata=[]
for item in fetch['sources']:
    with tarfile.open(r/item['archive_path'],'r:*') as archive:metadata.append(json.loads(archive.extractfile('00README.json').read()))
extra={k:metadata[1][k] for k in set(metadata[1])-set(metadata[0])}
assert extra=={'spec_version':1,'texlive_version':'2025'}
assert metadata[0]=={k:v for k,v in metadata[1].items() if k not in extra}
receipt=dict(all_checks_passed=True,script_sha256=sha(Path(__file__).read_bytes()),fetch_receipt_sha256=sha((run/'stage66_paper_source_fetch.json').read_bytes()),
    versions=[4,5],sole_manuscript_change='Abstract adds the public source-code repository URL',inserted_characters=len(addition),
    manuscript_identical_after_removing_code_link=True,case_study_section_identical=True,case_study_section_sha256=sha(sections[0].encode()),
    changed_archive_members=changed_members,added_build_metadata=extra,unchanged_archive_members=len(set(members[0])&set(members[1]))-2,
    journal_pdf_equivalence_established=False,author_code_numerical_equivalence_established=False,
    exact_dataset_manifest_established=False,cheng_finetune_recipe_established=False,
    interpretation='arXiv v4/v5 case-study text and figures agree; build metadata adds spec/TeX Live versions. No equivalence to journal PDF or execution/checkpoint is inferred')
(run/'stage66_paper_version_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
