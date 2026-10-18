"""Consolidate root-level README_MILESTONE*.md into one docs file.

The repository README.md remains the project entry point. Original milestone
notes are concatenated verbatim with source names/hashes for provenance.
Dry-run by default; --apply writes the consolidated document and README link
before unlinking only the listed root-level milestone files.

Usage (from project root):
    python scripts/consolidate_neural_readmes.py
    python scripts/consolidate_neural_readmes.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile

PATTERN = re.compile(r'README_MILESTONE([1-9][0-9]*)\.md')
SECTION_TITLE = '### Revised SOLETE neural development-CV track'
DOC_RELATIVE = Path('docs/revised_neural_milestones.md')
INDEX_TEXT = '''### Revised SOLETE neural development-CV track

The revised SOLETE neural experiment is maintained separately from the historical
executed record in `notebooks/11_deep_learning_models.ipynb`:

- Current development notebook: `notebooks/11_revised_neural_models.ipynb`.
- LSTM/TCN nested expanding-window CV: 15/30/60-minute horizons, F0–F4
  predictor families, four frozen candidate configurations per architecture,
  and three outer folds (360 candidate-fold evaluations in total).
- The original source-fingerprinted F0 results remain under
  `results/development_only/revised_neural_cv_v1/`. New F1–F4 results use the
  distinct, versioned `revised_neural_cv_v2_year_std1/` archive to handle
  a training-constant `valid_utc_year` without dividing by zero. These
  scaling policies and their source fingerprints must be reported separately.
- **Research status:** revised development selection is in progress; no revised
  terminal-holdout scoring is part of the ongoing campaign.
- Full chronological implementation and decisions: [Neural milestones 1–6](docs/revised_neural_milestones.md).

'''


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def consolidate(root: Path, *, apply: bool = False) -> dict:
    root = Path(root).resolve()
    readme = root / 'README.md'
    if not readme.is_file():
        raise FileNotFoundError(f'Missing project README: {readme}')
    originals = sorted(
        (p for p in root.glob('README_MILESTONE*.md') if PATTERN.fullmatch(p.name)),
        key=lambda p: int(PATTERN.fullmatch(p.name).group(1)),
    )
    target = root / DOC_RELATIVE
    old_index = readme.read_text(encoding='utf-8')
    if not originals:
        if target.is_file() and SECTION_TITLE in old_index:
            return {'status':'already consolidated', 'count':0, 'target':str(DOC_RELATIVE)}
        raise ValueError('No root-level milestone READMEs found (nothing to consolidate)')
    if target.exists():
        raise FileExistsError(f'Destination already exists; refusing overwrite: {target}')
    version_numbers = [int(PATTERN.fullmatch(p.name).group(1)) for p in originals]
    if len(version_numbers) != len(set(version_numbers)):
        raise ValueError('Duplicate milestone number')
    if min(version_numbers) != 1 or version_numbers != list(range(1, max(version_numbers) + 1)):
        raise ValueError(f'Missing intermediate milestone README: found {version_numbers}')
    if max(version_numbers) < 6:
        raise ValueError(f'Expected installed Milestone 6 README, found {version_numbers}')
    sections = [
        '# Revised SOLETE Neural CV — Milestone Notes\n\n'
        'This is a lossless consolidation of the former root-level '
        '`README_MILESTONEN.md` notes.\n'
        'Original source names and SHA-256 digests are retained for audit. '
        'See Git history for the independently committed originals.\n'
    ]
    for path in originals:
        number = int(PATTERN.fullmatch(path.name).group(1))
        content = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        sections.append(
            f'\n---\n\n## Milestone {number} — original `{path.name}`\n\n'
            f'Source SHA-256: `{digest}`\n\n' + content.rstrip('\n') + '\n'
        )
    text = '\n'.join(sections)
    # Confirm the concatenation still contains each original (no shortening).
    for path in originals:
        if path.read_text(encoding='utf-8').rstrip('\n') not in text:
            raise RuntimeError(f'Missing original content in merged document: {path.name}')
    if SECTION_TITLE not in old_index:
        if '## Models\n' in old_index:
            index = old_index.replace('## Models\n',INDEX_TEXT+'## Models\n',1)
        else:
            index = old_index.rstrip('\n')+'\n\n'+INDEX_TEXT
    elif 'docs/revised_neural_milestones.md' not in old_index:
        raise ValueError('Existing neural section lacks expected destination; manual review required')
    else:
        index = old_index
    details = {'status': 'dry run' if not apply else 'consolidated',
               'count':len(originals),'sources':[p.name for p in originals],
               'target':str(DOC_RELATIVE), 'bytes':len(text.encode('utf-8'))}
    if apply:
        _atomic_write(target, text)
        _atomic_write(readme, index)
        # Only remove the content after both replacement files were written.
        for path in originals:
            path.unlink()
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Perform verified consolidation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = consolidate(root, apply=args.apply)
    print('NEURAL README CONSOLIDATION:', result['status'].upper())
    for key, value in result.items():
        if key != 'status':
            print(f'{key}: {value}')
    if not args.apply and result['status'] != 'already consolidated':
        print('Review and rerun with --apply to write docs and remove root milestone notes.')


if __name__ == '__main__':
    main()
