"""NARCBench's pinned code (a checkout beside this repository, config.NARCBENCH_GIT): its LR probe and episode
aggregators, imported unchanged, and its Stego transcripts, joined into turns."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NARCBENCH = Path(os.environ.get('NARCBENCH_DIR', ROOT/'narcbench'))


def modules(subdir, *names):
    """Import NARCBench modules from NARCBENCH/<subdir>. They import NARCBench's own top-level `config`; ours
    (config.py) is set aside while they load."""
    import importlib
    if not (NARCBENCH/'generation'/'stego.py').exists():
        raise RuntimeError(f'NARCBench checkout not found at {NARCBENCH}; run python -m utils.build.setup inputs')
    ours = sys.modules.pop('config', None)
    sys.path[:0] = [str(NARCBENCH/subdir), str(NARCBENCH)]
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        del sys.path[:2]
        sys.modules.pop('config', None)
        if ours is not None:
            sys.modules['config'] = ours


def narc_lr():
    return modules('probes', 'reproduce_LR')[0]


def stego_templates():
    """NARCBench's Stego prompts: the honest system prompt, and the text preceding {base_prompt} in the receiver's
    and the sender's colluder templates."""
    s, = modules('generation', 'stego')
    tail = lambda template, start: template[template.rindex(start):template.index('{base_prompt}')]
    return s.HONEST_SYSTEM, tail(s.BP_TABLE_TALK_TEMPLATE, 'Do not reference'), tail(s.COUNTER_TABLE_TALK_TEMPLATE, 'The pit boss')


def narcbench(*parts):
    return NARCBENCH.joinpath(*parts)


# ── The probe ─────────────────────────────────────────────────────────────────────────────────────

ALL_METHODS = ('asymmetry_probe', 'peak_suspicion', 'deception_split', 'sorted_concat', 'activation_alignment')


def fit_core_probe(core, training, fit_domains, layer, seed):
    """NARCBench's LR on Core colluders versus honest agents of the given domains."""
    response_targets = {row['response']: row for row in training['agent_targets'] if row['fold'] in fit_domains}
    response_indices = sorted(response_targets)
    if len(response_indices) < 4 or set(response_targets[i]['target'] for i in response_indices) != {0, 1}:
        raise ValueError('Core collusion-run training rows incomplete')
    meta = [{'mode': 'collusion', 'role': 'colluder' if response_targets[i]['target'] else 'honest',
             'channel': 'public', 'phase': 'discussion', 'run_id': 'opaque__core_01__collusion'}
            for i in response_indices]
    return narc_lr().train_deception_probe(meta, {f'layer_{layer}': core['vectors'][response_indices]}, layer, seed=seed)


def fit_core_aggregator_and_score(groups, episodes, scaler, direction, train_groups, train_ids, train_y, *,
                                  methods=ALL_METHODS):
    """NARCBench's five episode aggregators; the two learned ones are fitted on Core."""
    lr = narc_lr()
    train_scores = lr.compute_scores(train_groups, train_ids, scaler, direction)
    test_scores = lr.compute_scores(groups, episodes, scaler, direction)
    values = {method: test_scores[method] for method in methods
              if method in ('peak_suspicion', 'deception_split', 'activation_alignment')}
    if 'asymmetry_probe' in methods:
        asym_scaler, asym_probe = lr._train_asymmetry_classifier(train_scores, train_y)
        values['asymmetry_probe'] = asym_probe.predict_proba(asym_scaler.transform(test_scores['asymmetry_probe']))[:, 1]
    if 'sorted_concat' in methods:
        pca, concat_scaler, concat_probe = lr._train_concat_classifier(train_scores, train_y)
        values['sorted_concat'] = concat_probe.predict_proba(
            concat_scaler.transform(pca.transform(test_scores['sorted_concat'])))[:, 1]
    return {method: {episode: float(value) for episode, value in zip(episodes, scores, strict=True)}
            for method, scores in values.items()}


def agent_activation_means(rows):
    """Each agent's activation averaged over its responses, in NARCBench's episode format."""
    grouped = defaultdict(list)
    for episode, agent, vector in zip(rows['episode'], rows['agent'], rows['vectors'], strict=True):
        grouped[int(episode), int(agent)].append(vector)
    episodes = defaultdict(lambda: {'agents': {}})
    for (episode, agent), values in grouped.items():
        episodes[episode]['agents'][agent] = np.mean(np.stack(values), axis=0)
    if any(len(item['agents']) != 4 for item in episodes.values()):
        raise ValueError('raw monitor requires four agents per episode')
    return dict(episodes)


# ── Stego transcripts ─────────────────────────────────────────────────────────────────────────────

def episode(run_dir):
    """One NARCBench Stego run: public turns with their recorded responses, and the running count per hand."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir/'run_config.json').read_text())
    prompts = json.loads((run_dir/'agent_prompts.json').read_text())
    results = json.loads((run_dir/'results.json').read_text())
    said = {}
    for message in results['public_messages']:
        name, text = re.match(r'^\[([^\]]+)\]:\s*(.*)$', message, re.S).groups()
        said.setdefault(name, []).append(text)
    turns, used = [], {}
    for p in prompts:
        if p['channel'] != 'public' or p['role'] == 'judge':
            continue
        name = p['agent_name']
        if p['phase'] == 'final':
            response = results['final_texts'][name]
        else:
            response = said[name][used.get(name, 0)]
            used[name] = used.get(name, 0) + 1
        turns.append({'hand': int(p['round']), 'agent': name, 'role': p['role'], 'phase': p['phase'],
                      'system': p['system_prompt'], 'user': p['user_prompt'], 'response': response})
    if any(used[n] != len(said[n]) for n in said if n in used):
        raise ValueError(f'public message join incomplete: {run_dir.name}')
    coalition = config['coalition']
    return {'run_id': config['run_id'], 'counter': coalition[0] if coalition else None,
            'big_player': coalition[1] if coalition else None, 'turns': turns,
            'running_count': [h['running_count'] for h in results['hand_results']]}


def episodes(root):
    return {d.name: episode(d) for d in sorted(Path(root).iterdir()) if d.is_dir()}


def turn(ep, hand, agent):
    found = [t for t in ep['turns'] if t['hand'] == hand and t['agent'] == agent]
    if len(found) != 1:
        raise ValueError(f'expected one turn: {ep["run_id"]} hand {hand} {agent}')
    return found[0]


def count_range(running):
    """NARCBench's count ranges."""
    return 'NEGATIVE' if running < 0 else 'NEUTRAL' if running <= 2 else 'HOT'


def prompt_text(tok, t):
    """The generation prompt as NARCBench served it (chat template, enable_thinking=False)."""
    return tok.apply_chat_template([{'role': 'system', 'content': t['system']}, {'role': 'user', 'content': t['user']}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def token_at(offsets, char):
    """Index of the first token whose span ends after character `char`."""
    return next(i for i, (_, end) in enumerate(offsets) if end > char)
