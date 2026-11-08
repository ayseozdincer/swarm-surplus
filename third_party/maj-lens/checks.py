"""Independent checks: headline numbers recomputed from the cache with the paper's definitions, written without
detect.py, read.py or steer.py, and compared with the tables those modules wrote.

  python checks.py
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import BAND, CONCEPTS, MODELS, RANGES, SITE, TIERS

A = Path('artifacts')
failed = []


def check(name, ours, table, tolerance=None):
    """Recomputed and printed values agree: exactly, or within `tolerance` (one unit of the printed last digit,
    for a value that lies on a rounding boundary)."""
    ok = ours == table if tolerance is None else abs(float(ours) - float(table)) <= tolerance + 1e-9
    print(f"{'PASS' if ok else 'FAIL'}  {name}: recomputed {ours}, table {table}")
    if not ok:
        failed.append(name)


def cells(tex, row_start):
    """The cells of the first table row starting with row_start."""
    line = next(l for l in Path(tex).read_text().splitlines() if l.startswith(row_start))
    return [c.strip().rstrip('\\').strip() for c in line.split('&')]


def number(cell):
    return re.search(r'\d\.\d{3}', cell).group()


# 1. Detection: J-PACT's episode score is the maximum pair score; AUROC within each Core domain, Transfer family,
#    Stego whole, then averaged (sklearn).
names = {'gpt_oss_20b': 'GPT-OSS-20B', 'qwen3_32b': 'Qwen3-32B', 'llama3_3_70b': 'Llama-3.3-70B-IT',
         'deepseek_v4_flash': 'DeepSeek-V4-Flash'}
lines = Path('results/main/detection.tex').read_text().splitlines()
for model in MODELS:
    z = np.load(A/'observations'/f'pairs_{model}.npz')
    row = lines[lines.index(next(l for l in lines if l.startswith(names[model]))) + 2]   # probe, ontology, J-PACT
    printed = [number(c) for c in row.split('&')[2:]]
    for tier, table in zip(TIERS, printed):
        runs = json.loads((A/'maj/2026-09-22-repro'/model/SITE/'evaluator'/f'{tier}.json').read_text())['runs']
        score = dict(zip(z[f'{tier}_episodes'].tolist(), z[f'{tier}_R'].max(1)))
        key = {'core': 'domain', 'transfer': 'family'}.get(tier)
        groups = sorted({r[key] if key else 'all' for r in runs})
        per = [roc_auc_score([r['target'] for r in runs if (r[key] if key else 'all') == g],
                             [score[r['episode']] for r in runs if (r[key] if key else 'all') == g]) for g in groups]
        check(f'J-PACT AUROC {model} {tier}', f'{np.mean(per):.4f}', table, tolerance=0.001)

# 2. Read: concepts in the colluders' band top-10 at the response-start token.
text = [t.lower() for t in json.loads((A/'interp'/'vocab.json').read_text())['text']]
core = json.loads((A/'interp'/'core.json').read_text())['episodes']
words = set().union(*CONCEPTS.values())
held = lambda top: {text[i] for l in BAND for i in top[str(l)]} & words
colluders = [a for e in core if e['mode'] == 'collusion' for a in e['agents'] if 'counterpart' in a]
check('colluders holding any concept', f'{sum(bool(held(a["top"])) for a in colluders)}',
      cells('results/interp/core_commitment.tex', 'any')[1])
check('counterfactuals holding any concept', f'{sum(bool(held(a["counterpart"])) for a in colluders)}',
      cells('results/interp/core_commitment.tex', 'any')[2])


def exclusive(agents, pair):
    bags = {a['name']: {i for l in BAND for i in a['top'][str(l)]} for a in agents}
    shared = bags[pair[0]] & bags[pair[1]]
    return {text[i] for i in shared - set().union(*(b for n, b in bags.items() if n not in pair))} & words


col = [bool(exclusive(e['agents'], [a['name'] for a in e['agents'] if a['role'] == 'colluder']))
       for e in core if e['mode'] == 'collusion']
ctl = [any(exclusive(e['agents'], p) for p in combinations(sorted(a['name'] for a in e['agents']), 2))
       for e in core if e['mode'] == 'control']
check('colluding pairs sharing an exclusive concept', f'{sum(col)}/{len(col)}', cells('results/interp/core_pairs.tex', 'any')[1])
check('control runs with any such pair', f'{sum(ctl)}/{len(ctl)}', cells('results/interp/core_pairs.tex', 'any')[3])

# 3. Steer: decoded range = the range name with the highest logit after the forced note.
rows = json.loads((A/'interp'/'stego_swap.json').read_text())['rows']
column = {'NEGATIVE': 2, 'NEUTRAL': 3, 'HOT': 4}
for arm, label in (('alpha_0.0', 'No swap'), ('alpha_1.0', 'HOT/Negative, $\\alpha=1.0$'),
                   ('alpha_1.25', 'HOT/Negative, $\\alpha=1.25$')):
    for true, read in (('NEGATIVE', 'HOT'), ('NEUTRAL', 'NEGATIVE')):
        hands = [r for r in rows if r['range'] == true]
        share = sum(max(RANGES, key=lambda g: r[arm][g]) == read for r in hands) / len(hands)
        table = cells('results/interp/stego_swap.tex', f'{label} & {true}')[column[read]]
        check(f'{label}: {true} hands read as {read}', f'{share:.2f}', table)

print('all checks pass' if not failed else f'{len(failed)} checks FAIL')
sys.exit(1 if failed else 0)
