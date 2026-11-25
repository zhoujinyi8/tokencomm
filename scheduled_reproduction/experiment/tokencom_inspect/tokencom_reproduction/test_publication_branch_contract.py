"""Exercise exact Git publication synchronization with an isolated alternate branch."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify():
    source = Path(__file__).with_name('sync_github_worktree_from_api.py')
    workspace = source.parent / 'runs'
    with tempfile.TemporaryDirectory(prefix='stage59_git_contract_', dir=workspace) as temporary:
        base = Path(temporary).resolve()
        assert base.parent == workspace.resolve()
        repository = base/'origin'; repository.mkdir()
        checkout = base/'work/tokencomm_github_sync'
        experiment = base/'work/tokencom_inspect/tokencom_reproduction'
        experiment.mkdir(parents=True)
        (experiment/'runs').mkdir()
        def git(*args, cwd=repository, env=None, data=None):
            return subprocess.check_output(['git', *args], cwd=cwd, env=env, input=data,
                                           text=True, encoding='utf-8', stderr=subprocess.PIPE).strip()
        git('init', '-b', 'main')
        git('config', 'user.name', 'Publication Contract Fixture')
        git('config', 'user.email', 'fixture@example.invalid')
        git('config', 'core.autocrlf', 'false')
        extras = ['.flake8', 'pytest.ini', 'requirements-ci.txt', '.github/workflows/python-app.yml',
                  'conftest.py', 'test_context_maskgit_v4.py', 'CI_TESTING.md',
                  'third_party/perceptual_similarity/lpips/trainer.py']
        for name in extras+['scheduled_reproduction/example.txt']:
            path = repository/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n', encoding='utf-8')
        git('add', '.'); git('commit', '-m', 'Fixture parent')
        parent = git('rev-parse', 'HEAD')
        branch = 'reproduction/alternate-contract-fixture'
        git('worktree', 'add', '-b', branch, str(checkout), parent)
        git('update-ref', 'refs/remotes/origin/'+branch, parent, cwd=checkout)
        (checkout/'scheduled_reproduction/example.txt').write_text('next fixture\n', encoding='utf-8')
        git('add', 'scheduled_reproduction', cwd=checkout)
        tree = git('write-tree', cwd=checkout)
        env = dict(os.environ, GIT_AUTHOR_DATE='2026-11-26T04:13:00+08:00',
                   GIT_COMMITTER_DATE='2026-11-26T04:13:00+08:00')
        commit = git('commit-tree', tree, '-p', parent, cwd=checkout, env=env, data='Fixture publication\n')
        person = {'name':'Publication Contract Fixture', 'email':'fixture@example.invalid',
                  'date':'2026-11-25T20:13:00Z'}
        metadata = {'sha':commit, 'tree':{'sha':tree}, 'parents':[{'sha':parent}],
                    'message':'Fixture publication', 'author':person, 'committer':person}
        (experiment/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        helper = experiment/source.name; shutil.copy2(source, helper)
        command = [sys.executable, str(helper), '--commit', commit, '--expected-parent', parent,
                   '--metadata', 'metadata.json', '--receipt', 'runs/receipt.json']
        def execute(selected_branch):
            return subprocess.run(command+['--branch', selected_branch], cwd=experiment,
                                  capture_output=True, text=True, timeout=30)
        wrong = execute('reproduction/wrong-fixture')
        assert wrong.returncode != 0 and git('rev-parse', 'HEAD', cwd=checkout) == parent
        first = execute(branch)
        assert first.returncode == 0, first.stderr
        receipt = json.loads((experiment/'runs/receipt.json').read_text(encoding='utf-8'))
        assert receipt['publication_branch'] == branch and receipt['head_after'] == commit
        assert receipt['exact_commit_object_verified'] and receipt['tree_matches_remote_api'] and receipt['worktree_clean']
        assert git('rev-parse', 'refs/remotes/origin/'+branch, cwd=checkout) == commit
        second = execute(branch)
        assert second.returncode == 0, second.stderr
        assert git('rev-parse', 'HEAD', cwd=checkout) == commit
        (checkout/'scheduled_reproduction/example.txt').write_text('incorrect tree\n', encoding='utf-8')
        mismatch = execute(branch)
        assert mismatch.returncode != 0 and git('rev-parse', 'HEAD', cwd=checkout) == commit
        return {'passed':4, 'failed':0, 'network_used':False, 'model_inference_performed':False,
                'checks':['wrong branch refused before ref mutation', 'alternate branch exact commit/tree',
                          'idempotent rerun', 'tree mismatch refuses ref mutation'],
                'helper_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'test_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    result = verify()
    target = Path(__file__).parent/'runs/stage59_publication_branch_contract.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
