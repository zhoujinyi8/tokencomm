"""Record scoped primary-source observations and reject relabeling small cohorts as ImageNet100."""
import ast,hashlib,json,subprocess
from pathlib import Path
from imagenet100_manifest import validate
r=Path(__file__).resolve().parent;run=r/'runs';official=r.parents[1]/'tokencom_official_reference'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
remote=json.loads((run/'stage65_remote_source_check.json').read_text(encoding='utf-8'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()==remote['observed_head_sha']
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=official,text=True).strip()
mapping=(official/'synset_words.txt').read_text(encoding='utf-8').splitlines();synsets=[line.split()[0] for line in mapping]
module=ast.parse((r/'evaluate_stage9_pretrained_maskgit.py').read_text(encoding='utf-8'))
class_ids=ast.literal_eval(next(n.value for n in ast.walk(module) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMAGENET_CLASS_IDS' for t in n.targets)))
diagnostic=json.loads((run/'stage65_cheng_diagnostic_plan.json').read_text(encoding='utf-8'))
development=sorted({synsets[class_ids[v['class_name']]] for v in diagnostic['images']})
author=json.loads((run/'stage64_author_dataset_protocol.json').read_text(encoding='utf-8'));demo=[synsets[i] for i in author['config']['class_indices']]
probes=[]
for name,classes in [('existing ten-class development cohort',development),('pinned demo default',demo)]:
    try:validate(dict(classes=classes,images=[],class_list_source='source inspection',split_source='not established'),set(synsets))
    except ValueError as exc:probes.append(dict(candidate=name,class_count=len(classes),rejected=True,reason=str(exc)))
    else:raise AssertionError('Small cohort was accepted as ImageNet100')
book=(official/'TokCom_Demo.ipynb').read_bytes();raw_blob=hashlib.sha1(b'blob '+str(len(book)).encode()+b'\0'+book).hexdigest()
blob=subprocess.check_output(['git','rev-parse','HEAD:TokCom_Demo.ipynb'],cwd=official,text=True).strip()
parent='bd3f2808123fd0464040ffc8acf1851b333f67f6';checkout=r.parents[1]/'tokencomm_github_sync'
published=json.loads(subprocess.check_output(['git','show',parent+':scheduled_reproduction/SNAPSHOT_SHA256.json'],cwd=checkout,text=True,encoding='utf-8'))
legacy={}
for name in ('stage44_frozen_protocol.json','stage47_complete_holdout_summary.json','stage59_development_recovery_audit.json','stage58_session_comparison_audit.json','stage64_session_audit.json','stage64_matched_arq_comparison.json'):
    h=sha(run/name);assert h==published['evidence/'+name];legacy[name]=h
result=dict(all_checks_passed=True,script_sha256=sha(__file__),manifest_helper_sha256=sha(r/'imagenet100_manifest.py'),
    remote_check_sha256=sha(run/'stage65_remote_source_check.json'),current_notebook_git_blob=blob,checkout_notebook_raw_blob_sha=raw_blob,
    notebook_identical_to_first_public_release=blob==remote['initial_public_release']['git_blob_sha'],
    small_cohort_rejection_probes=probes,actual_paper_dataset_files_verified=0,paper_dataset_identity_established=False,
    paper_cheng_requires_finetuned_about_004_bpp=True,stock_quality_parameter_is_not_a_bitrate_guarantee=True,
    unresolved=['Exact paper 100-class synsets and image/split identities','Cheng variant, fine-tuned checkpoint, training data/loss/steps','Complete feedback/header/CMI and error-detection assumptions','Journal/arXiv/code version alignment'],
    parent_commit=parent,legacy_artifact_sha256_unchanged=legacy,external_messages_sent=False,full_paper_reproduction_complete=False)
(run/'stage65_protocol_gaps.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
