"""Restore pinned upstream sources, then apply the project's source overlays."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-workspace', type=Path, help='Use existing local git repositories under WORKSPACE/src; no network')
    args = parser.parse_args()
    locks = json.loads((ROOT / 'configs/upstream-lock.json').read_text())
    for name, directory in [('lingbot', 'lingbot-vla-v2'), ('robotwin', 'RoboTwin')]:
        dest = ROOT / 'src' / directory
        if dest.exists():
            raise SystemExit(f'Refusing to overwrite existing source directory: {dest}')
        dest.mkdir(parents=True)
        lock = locks[name]
        if args.from_workspace:
            archive = subprocess.check_output(['git', '-C', str(args.from_workspace / 'src' / directory), 'archive', lock['commit']])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
                tf.extractall(dest, filter='data')
        else:
            repo = lock['url'].removesuffix('.git')
            url = repo.replace('https://github.com/', 'https://codeload.github.com/') + '/tar.gz/' + lock['commit']
            archive = subprocess.check_output(['curl', '--fail', '--location', '--silent', '--show-error', '--limit-rate', '128K', url])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
                members = tf.getmembers()
                prefix = members[0].name.split('/')[0] + '/'
                for item in members:
                    if not item.name.startswith(prefix) or item.name == prefix:
                        continue
                    item.name = item.name[len(prefix):]
                    if item.name:
                        tf.extract(item, dest, filter='data')
        overlay = ROOT / 'overlays' / directory
        if overlay.exists():
            shutil.copytree(overlay, dest, dirs_exist_ok=True)
        (dest / '.project-upstream.json').write_text(json.dumps(lock, indent=2))
        print(f'{directory}: restored {lock["commit"]}')
    print('Install the upstream training and simulator environments; see docs/REPRODUCIBILITY.md.')

if __name__ == '__main__':
    main()
