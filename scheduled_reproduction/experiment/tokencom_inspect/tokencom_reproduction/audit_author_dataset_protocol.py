"""Read-only evidence for the pinned author's demo dataset configuration."""
import ast,hashlib,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;official=r.parents[1]/'tokencom_official_reference'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()
assert head=='0422ed32dcde3347e0bb6697f343f7a9ff7b3214'
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=official,text=True).strip()
book=json.loads((official/'TokCom_Demo.ipynb').read_text(encoding='utf-8'))
config=''.join(book['cells'][6]['source']);loader=''.join(book['cells'][8]['source'])
fields={}
for node in ast.walk(ast.parse(config)):
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self' and target.attr in ('class_indices','num_images','seed','snr_list','num_simulations'):
                fields[target.attr]=ast.literal_eval(node.value)
assert fields['class_indices']==[503,498] and fields['num_images']==40
assert 'os.listdir(folder_path)' in loader and 'random.sample(images, num_images)' in loader
assert 'sorted(' not in loader
root_assignment=next(n for n in ast.walk(ast.parse(loader)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='directory_path' for t in n.targets))
classes=(official/'synset_words.txt').read_text(encoding='utf-8').splitlines()
result=dict(all_checks_passed=True,scope='Pinned demo source inspection only; not proof of the paper evaluation subset',
    official_commit=head,script_sha256=sha(Path(__file__)),source_sha256={n:sha(official/n) for n in ('README.md','TokCom_Demo.ipynb','synset_words.txt')},
    notebook_cell_indices=dict(config=6,loader=8),config=fields,configured_class_names={str(i):classes[i] for i in fields['class_indices']},
    configured_demo_images=fields['num_images']*len(fields['class_indices']),
    loader_root=ast.literal_eval(root_assignment.value),config_root_uses_imagenet_root_environment='os.environ.get("IMAGENET_ROOT"' in config,
    loader_uses_separate_literal_root=True,sampling='random.sample over unsorted os.listdir; global config seed is 1',
    exact_paper_100_class_manifest_established=False,exact_paper_image_identities_established=False,
    next_required_evidence=['Author paper 100-class synset list','Evaluation split and exact image IDs/hashes','Sampling order and seeds','Cheng configuration/checkpoint and complete forward/feedback/CMI resource protocol'],
    official_source_changed=False,new_data_downloaded=False,new_model_inference=False,full_paper_reproduction_complete=False)
(r/'runs/stage64_author_dataset_protocol.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
