"""Read: what DeepSeek-V4-Flash's colluders hold in their J-Space (workspace band L19-38), on transcripts DeepSeek
wrote itself with NARCBench's generation code (utils/generate_cohort.sh).

  band         Anthropic's workspace criteria per layer: does the top-10 readout contain the model's next token,
               and does its top-1 persist across positions above a shuffled null
  core         every agent of every Core episode at its response-start token, all lens layers; each colluder
               also through its honest counterfactual (same prompt, the table's honest brief, its own name)
  transcripts  three colluders write their reply; the readout at every generated token, colluder vs its
               counterfactual reading the same tokens
  stego        the Stego receiver at the last token of the sender's table talk and at its response start, and
               its honest counterfactual (NARCBench's honest prompt, same table and chat)

"Holds" a token: the token is in the top-10 J-lens readout at some band layer at that position.

  python read.py    collect any stage missing from the cache (GPU), then write the Read tables and figures
"""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import binomtest, fisher_exact

from cache import SCENARIOS
from config import BAND, BAND_POSITIONS, BAND_PROMPTS, BAND_SEED, CONCEPTS, GEN_TOKENS, LENS_LAYERS, RANGE_SIGNATURES, \
    RANGES, TOPK, TRANSCRIPT_RUNS
from lens import Readout
from utils import deepseek as DS, narcbench as NB
from utils.interp import finals, load_stage, vocab

CORE = SCENARIOS/'deepseek_v4_flash'/'core'
STEGO = SCENARIOS/'deepseek_v4_flash'/'stego'
VOCAB = set().union(*CONCEPTS.values())
GROUPS = {**CONCEPTS, 'any': VOCAB}
PRIVATE = re.compile(r'^[ \t]*PRIVATE\s*:(?!\s*\[what).*\n?', flags=re.M)   # the receiver's private decode line


# ═══════════════════════════════════════════════════════════════════ Sites and counterfactuals

def encode(tok, turn, prefix=True):
    """Chat prompt of one turn; with `prefix`, ending at the response-start token (the reply's first position)."""
    text = NB.prompt_text(tok, turn)
    if prefix and not text.endswith(DS.PUBLIC_RESPONSE_PREFIX):
        text += DS.PUBLIC_RESPONSE_PREFIX
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc['input_ids'], enc['offset_mapping']


def strip_private(s):
    return PRIVATE.sub('', s)


def core_counterfactual(p, honest):
    """A Core colluder's honest counterfactual: its user prompt, the table's honest brief, its own name."""
    name, system = honest
    return {'system': system.replace(name, p['agent_name']), 'user': p['user_prompt']}


def stego_counterfactual(t, partner, receiver=True):
    """A Stego colluder's honest counterfactual: same agent, table and chat; NARCBench's honest system prompt and
    plain table prompt."""
    honest, receiver_tail, sender_tail = NB.stego_templates()
    tail = (receiver_tail if receiver else sender_tail).format(partner=partner)
    if t['user'].count(tail) != 1:
        raise ValueError('colluder template tail not found exactly once')
    return {**t, 'system': honest.format(name=t['agent']), 'user': t['user'].split(tail, 1)[1]}


# ═══════════════════════════════════════════════════════════════════ Collection (GPU)

def collect_band(model, tok, spec):
    lens, rng = Readout(model, spec, LENS_LAYERS), np.random.default_rng(BAND_SEED)
    rows = []
    for run in sorted(p for p in CORE.iterdir() if p.name.endswith('__collusion'))[:BAND_PROMPTS]:
        p = finals(run)[0]
        _, ids, _ = encode(tok, {'system': p['system_prompt'], 'user': p['user_prompt']}, prefix=False)
        nxt = DS.forward(model, ids, all_logits=True).logits[0].float().argmax(-1).cpu().numpy()
        pos = np.sort(rng.choice(np.arange(20, len(ids) - 2), BAND_POSITIONS, replace=False))
        row = {'run': run.name, 'acc': [], 'auto': [], 'auto_null': []}
        for l in lens.layers:
            top10 = lens.logits(l, pos).topk(TOPK, dim=-1).indices.cpu().numpy()
            row['acc'].append(float((top10 == nxt[pos][:, None]).any(-1).mean()))
            top1 = top10[:, 0]
            row['auto'].append(float(np.mean(top1[1:] == top1[:-1])))
            null = top1[rng.permutation(len(top1))]                  # the same sequence, positions shuffled
            row['auto_null'].append(float(np.mean(null[1:] == null[:-1])))
        rows.append(row)
    lens.close()
    return {'layers': list(lens.layers), 'runs': rows}


def collect_core(model, tok, spec):
    lens, episodes = Readout(model, spec, LENS_LAYERS), []
    for run in sorted(p for p in CORE.iterdir() if p.is_dir()):
        fs = finals(run)
        honest = next(((p['agent_name'], p['system_prompt']) for p in fs if p['role'] == 'honest'), None)
        agents = []
        for p in fs:
            _, ids, _ = encode(tok, {'system': p['system_prompt'], 'user': p['user_prompt']})
            DS.forward(model, ids)
            a = {'name': p['agent_name'], 'role': p['role'], 'top': {l: v[0] for l, v in lens.top([len(ids) - 1]).items()}}
            if p['role'] == 'colluder' and honest:
                _, ids, _ = encode(tok, core_counterfactual(p, honest))
                DS.forward(model, ids)
                a['counterpart'] = {l: v[0] for l, v in lens.top([len(ids) - 1]).items()}
            agents.append(a)
        episodes.append({'run': run.name, 'mode': run.name.rsplit('__', 1)[1], 'agents': agents})
        print('core', run.name, flush=True)
    lens.close()
    return {'episodes': episodes}


def collect_transcripts(model, tok, spec):
    lens, rows = Readout(model, spec, BAND), []
    for run in (CORE/r for r in TRANSCRIPT_RUNS):
        fs = finals(run)
        honest = next(((p['agent_name'], p['system_prompt']) for p in fs if p['role'] == 'honest'), None)
        for p in (p for p in fs if p['role'] == 'colluder' and honest):
            _, pc, _ = encode(tok, {'system': p['system_prompt'], 'user': p['user_prompt']})
            _, ph, _ = encode(tok, core_counterfactual(p, honest))
            gen = DS.generate(model, pc, tok.eos_token_id, GEN_TOKENS)
            DS.forward(model, pc + gen); col = lens.top(list(range(len(pc) - 1, len(pc) + len(gen))))
            DS.forward(model, ph + gen); hon = lens.top(list(range(len(ph) - 1, len(ph) + len(gen))))
            rows.append({'run': run.name, 'agent': p['agent_name'], 'system': p['system_prompt'], 'generated': gen,
                         'colluder': col, 'counterpart': hon})
        print('transcripts', run.name, flush=True)
    lens.close()
    return {'rows': rows}


def collect_stego(model, tok, spec):
    """Every hand of every Stego collusion episode, PRIVATE: lines stripped from every prompt."""
    lens, rows = Readout(model, spec, BAND), []
    for run, ep in NB.episodes(STEGO).items():
        if not run.endswith('collusion'):
            continue
        snd, rcv = ep['counter'], ep['big_player']
        for hand in range(1, 8):
            ts, tr = NB.turn(ep, hand, snd), NB.turn(ep, hand, rcv)
            msg = strip_private(ts['response']).strip()
            reads = {}
            for key, turn, at_msg in (('receiver', tr, True), ('receiver_honest', stego_counterfactual(tr, snd, True), True),
                                      ('sender', ts, False), ('sender_honest', stego_counterfactual(ts, rcv, False), False)):
                text, ids, offs = encode(tok, {**turn, 'user': strip_private(turn['user'])})
                pos = {'start': len(ids) - 1}
                i, tail = text.rfind(f'[{snd}]:'), msg[-40:]
                j = text.find(tail, i) if i >= 0 else -1
                if at_msg and j >= 0:
                    pos['message_end'] = NB.token_at(offs, j + len(tail) - 1)
                DS.forward(model, ids)
                reads[key] = {k: {l: v[0] for l, v in lens.top([q]).items()} for k, q in pos.items()}
            rows.append({'run': run, 'hand': hand, 'range': NB.count_range(ep['running_count'][hand - 1]),
                         'sender': snd, 'receiver': rcv, 'message': msg, **reads})
        print('stego', run, flush=True)
    lens.close()
    return {'rows': rows}


# ═══════════════════════════════════════════════════════════════════ Numbers

def bag(top, layers=BAND):
    """Token ids in the top-10 at any of `layers` (one position)."""
    return {i for l in layers for i in top[str(l)]}


def named(ids, words):
    low = vocab()['low']
    return {low[i] for i in ids if low[i] in words}


def pval(p):
    return float(f'{p:.2g}')


def band_numbers(b):
    mean = lambda k: np.mean([r[k] for r in b['runs']], axis=0)
    return {'layers': b['layers'], 'next_token_top10': mean('acc').round(3).tolist(),
            'top1_persistence': mean('auto').round(3).tolist(), 'top1_persistence_null': mean('auto_null').round(3).tolist()}


def of_mode(episodes, mode='collusion'):
    return [e for e in episodes if e['mode'] == mode]


def commitment(core, words):
    """Each colluder vs its honest counterfactual: how many band tokens of `words` each holds at the response start."""
    low = vocab()['low']
    hits = lambda top: sum(low[i] in words for l in BAND for i in top[str(l)])
    rows = [{'colluder': hits(a['top']), 'counterpart': hits(a['counterpart'])}
            for e in of_mode(core['episodes']) for a in e['agents'] if 'counterpart' in a]
    more = sum(r['colluder'] > r['counterpart'] for r in rows)
    less = sum(r['colluder'] < r['counterpart'] for r in rows)
    return {'colluders': len(rows), 'colluder_more': more, 'counterpart_more': less,
            'sign_test_p': pval(binomtest(more, more + less).pvalue),
            'colluders_carrying': int(sum(r['colluder'] > 0 for r in rows)),
            'counterparts_carrying': int(sum(r['counterpart'] > 0 for r in rows))}


def shared_exclusive(bags, pair):
    """Tokens both agents of `pair` hold and no other agent at the table holds."""
    a, b = pair
    return (bags[a] & bags[b]) - set().union(*(v for k, v in bags.items() if k not in pair))


def pairs(core):
    """Per concept group: colluding pairs holding a shared-exclusive concept, the two honest agents of the same
    tables, and honest tables where any of the six pairs does."""
    out = {}
    for label, words in GROUPS.items():
        col, hon_pair, ctl_ep = [], [], []
        for e in of_mode(core['episodes']):
            bags = {a['name']: bag(a['top']) for a in e['agents']}
            c = tuple(a['name'] for a in e['agents'] if a['role'] == 'colluder')
            h = tuple(a['name'] for a in e['agents'] if a['role'] != 'colluder')
            col.append(bool(named(shared_exclusive(bags, c), words)))
            hon_pair.append(bool(named(shared_exclusive(bags, h), words)))
        for e in of_mode(core['episodes'], 'control'):
            bags = {a['name']: bag(a['top']) for a in e['agents']}
            names = sorted(bags)
            ctl_ep.append(any(bool(named(shared_exclusive(bags, (x, y)), words)) for i, x in enumerate(names) for y in names[i + 1:]))
        out[label] = {'colluding_pairs': [sum(col), len(col)], 'honest_pairs_same_tables': [sum(hon_pair), len(hon_pair)],
                      'control_episodes_any_pair': [sum(ctl_ep), len(ctl_ep)],
                      'fisher_p_vs_control_episodes': pval(fisher_exact([[sum(col), len(col) - sum(col)],
                                                                         [sum(ctl_ep), len(ctl_ep) - sum(ctl_ep)]], 'greater')[1])}
    return out


def pieces(ids):
    """Text of each generated token, by incremental UTF-8 decoding (a multi-byte character lands on its last token)."""
    raw, out, prev = b'', [], ''
    for i in ids:
        raw += bytes.fromhex(vocab()['bytes'][i])
        cur = raw.decode('utf-8', 'ignore'); out.append(cur[len(prev):]); prev = cur
    return out


def callouts(row, side, other):
    """(position, layer, concept): a concept of the groups in `side`'s top-10 at that token and layer, absent from
    `other`'s top-10 at the same token and layer, and not yet written."""
    low, text, found = vocab()['low'], pieces(row['generated']), []
    for k in range(1, len(row['generated']) + 1):
        written = ''.join(text[:k]).lower()
        for l in BAND:
            mine = {low[i] for i in row[side][str(l)][k] if low[i] in VOCAB}
            theirs = {low[i] for i in row[other][str(l)][k]}
            found += [(k, l, c) for c in sorted(mine - theirs) if c not in written]
    return found


def decoding(st):
    """Share of hands of each true range whose band readout at the end of the sender's message names each range
    (RANGE_SIGNATURES), for the colluding receiver and its honest counterfactual."""
    both = [r for r in st['rows'] if r['receiver'].get('message_end') and r['receiver_honest'].get('message_end')]
    low = vocab()['low']
    matrix = {}
    for who in ('receiver', 'receiver_honest'):
        got = [{g for g, ws in RANGE_SIGNATURES.items() if {low[i] for i in bag(r[who]['message_end'])} & ws} for r in both]
        n = {t: sum(r['range'] == t for r in both) for t in RANGES}
        matrix[who] = {t: {g: round(sum(g in x for x, r in zip(got, both) if r['range'] == t) / n[t], 2) for g in RANGES}
                       | {'hands': n[t]} for t in RANGES}
    return matrix


def report():
    from utils import figures, tables
    core, st, tr = load_stage('core', collect_core), load_stage('stego', collect_stego), load_stage('transcripts', collect_transcripts)
    com = {g: commitment(core, w) for g, w in GROUPS.items()}
    pr, dec = pairs(core), decoding(st)
    tables.interp_read(com, pr, dec)
    figures.band(band_numbers(load_stage('band', collect_band)))
    figures.quant(com, pr, dec)
    figures.stego(st)
    figures.transcript(tr, core)


if __name__ == '__main__':
    report()
