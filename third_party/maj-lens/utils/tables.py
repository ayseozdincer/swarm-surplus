"""Writing the LaTeX tables of results/. Every number arrives computed (detect.py, read.py, steer.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MODELS, NAMES, RANGES, SHOWN, TIERS

MAIN, INTERP = Path('results/main'), Path('results/interp')


def bracket(lo, hi, digits=2):
    return f'[{lo:.{digits}f}, {hi:.{digits}f}]'


def tabular(rows, header, fmt):
    text_columns = sum(1 for h in header if h in ('Model', 'Detector'))
    tex = ['\\begin{tabular}{' + 'l'*text_columns + 'r'*(len(header)-text_columns) + '}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    tex += [' & '.join(c.replace('%', '\\%').replace('**', '') for c in fmt(r)) + ' \\\\' for r in rows]
    tex += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(tex) + '\n'


def write(path, tex):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)


def detection(rows):
    """rows: {model, detector, core/transfer/stego: (AUROC, 95% interval)}. Bold: best detector per model and split;
    underlined: best unsupervised detector."""
    unsup = ('Ontology baseline', 'J-PACT')
    best, best_u = {}, {}
    for r in rows:
        for t in TIERS:
            v = round(r[t][0], 3)
            best[r['model'], t] = max(best.get((r['model'], t), -1.0), v)
            if r['detector'] in unsup:
                best_u[r['model'], t] = max(best_u.get((r['model'], t), -1.0), v)

    def cell(r, t):
        v = f'{r[t][0]:.3f}'
        if r['detector'] in unsup and round(r[t][0], 3) == best_u[r['model'], t]:
            v = f'\\underline{{{v}}}'
        if round(r[t][0], 3) == best[r['model'], t]:
            v = f'\\textbf{{{v}}}'
        return f'{v} {bracket(*r[t][1])}'
    name = lambda m: 'Four-model mean' if m == 'mean' else NAMES[m]
    tex = tabular(rows, ['Model', 'Detector', 'Core', 'Transfer', 'Stego'],
                  lambda r: [name(r['model']) if r['detector'] == 'NARCBench probe' else '', r['detector'],
                             *(cell(r, t) for t in TIERS)])
    write(MAIN/'detection.tex', '\\resizebox{\\linewidth}{!}{%\n' + tex + '}\n')


def observations(cells, columns):
    """cells[model or 'mean', (tier, n)] = (AUROC or None, 95% interval or None). Bold: the best count per model
    within Core and within Stego."""
    def row(key):
        best = {t: max((round(v, 3) for (m, (tt, _)), (v, _) in cells.items() if m == key and tt == t and v is not None),
                       default=None) for t in ('core', 'stego')}
        out = []
        for c in columns:
            v, ci = cells[key, c]
            text = '--' if v is None else f'{v:.3f}'
            if v is not None and best.get(c[0]) == round(v, 3):
                text = f'\\textbf{{{text}}}'
            out.append(text + (' ' + bracket(*ci) if ci else ''))
        return out
    groups = [('Core', len(SHOWN['core'])), ('Stego', len(SHOWN['stego'])), ('Transfer', len(SHOWN['transfer']))]
    counts = ['1' if tier == 'transfer' else str(n) for tier, n in columns]   # Transfer: one round per agent
    spans, start = [], 2
    for _, width in groups:
        spans.append(f'\\cmidrule(lr){{{start}-{start + width - 1}}}')
        start += width
    body = [' & '.join([NAMES[m]] + row(m)) + ' \\\\' for m in MODELS]
    body += ['\\midrule', ' & '.join(['Four-model mean'] + row('mean')) + ' \\\\']
    write(MAIN/'observations.tex', '\n'.join(['\\begin{tabular}{l' + 'r' * len(columns) + '}', '\\toprule',
                                              ' & ' + ' & '.join(f'\\multicolumn{{{w}}}{{c}}{{{g}}}' for g, w in groups) + ' \\\\',
                                              ' '.join(spans), 'Model & ' + ' & '.join(counts) + ' \\\\', '\\midrule']
                                             + body + ['\\bottomrule', '\\end{tabular}']) + '\n')


def localisation(accuracy):
    """accuracy[model][tier]: share of detected collusion episodes whose top pair splits along the coalition, or
    None (fewer than MIN_DETECTED detected)."""
    fmt = lambda v: '--' if v is None else f'{v:.3f}'
    mean = {t: [accuracy[m][t] for m in MODELS if accuracy[m][t] is not None] for t in TIERS}
    rows = [[NAMES[m], *(fmt(accuracy[m][t]) for t in TIERS)] for m in MODELS]
    rows.append(['Mean', *(fmt(float(np.mean(mean[t])) if mean[t] else None) for t in TIERS)])
    write(MAIN/'localisation.tex', tabular(rows, ['Model', 'Core', 'Transfer', 'Stego'], lambda r: r))


def small(name, header, rows):
    """A booktabs tabular scaled to the text width, first column left-aligned."""
    esc = lambda c: str(c).replace('%', '\\%').replace('_', '\\_')
    lines = ['\\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', '\\toprule', ' & '.join(map(esc, header)) + ' \\\\', '\\midrule']
    lines += [' & '.join(map(esc, r)) + ' \\\\' for r in rows]
    write(INTERP/f'{name}.tex', '\\resizebox{\\linewidth}{!}{%\n' + '\n'.join(lines + ['\\bottomrule', '\\end{tabular}']) + '}\n')


def interp_read(com, pr, dec):
    frac = lambda xy: f'{xy[0]}/{xy[1]}'
    small('core_commitment', ['Concept group', 'Colluders holding it', 'Counterfactuals holding it', 'Colluder more', 'Counterfactual more', 'Sign test $p$'],
          [[g, c['colluders_carrying'], c['counterparts_carrying'], c['colluder_more'], c['counterpart_more'], c['sign_test_p']]
           for g, c in com.items()])
    small('core_pairs', ['Concept group', 'Colluding pairs', 'Honest pairs, same runs', 'Control runs (any pair)', 'Fisher $p$ (runs)'],
          [[g, frac(v['colluding_pairs']), frac(v['honest_pairs_same_tables']), frac(v['control_episodes_any_pair']),
            v['fisher_p_vs_control_episodes']] for g, v in pr.items()])
    small('stego_decoding', ['Reader', 'True range (hands)', 'NEGATIVE', 'NEUTRAL', 'HOT'],
          [[lab, f"{t} ({dec[who][t]['hands']})", *(f"{dec[who][t][g]:.2f}" for g in RANGES)]
           for who, lab in (('receiver', 'Colluding receiver'), ('receiver_honest', 'Honest counterfactual')) for t in RANGES])


def interp_steer(sw):
    name = lambda k: ('No swap' if k == 'alpha_0.0' else f"HOT/Negative, $\\alpha={k.split('_')[1]}$" if k.startswith('alpha_')
                      else 'Random pairs, pooled' if k == 'random_pooled' else 'Random pair: ' + ' / '.join(w.strip() for w in sw['random_pairs'][int(k[7:])]))
    small('stego_swap', ['Swap', 'True range (hands)', 'NEGATIVE', 'NEUTRAL', 'HOT'],
          [[name(k), f"{t} ({m[t]['hands']})", *(f'{m[t][g]:.2f}' for g in RANGES)]
           for k, m in sw.items() if k != 'random_pairs' for t in RANGES])
