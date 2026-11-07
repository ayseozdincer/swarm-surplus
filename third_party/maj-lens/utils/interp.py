"""The DeepSeek interpretability cache: one JSON per stage under artifacts/interp/, collected on first use."""
from __future__ import annotations

import json

from cache import INTERP

_MODEL, _VOCAB = [], {}


def model():
    """(model, tokenizer, spec), loaded once."""
    if not _MODEL:
        from transformers import AutoTokenizer
        from utils import deepseek as DS
        from utils.models import get_model
        spec = get_model('deepseek_v4_flash')
        tok = DS.configure_tokenizer(AutoTokenizer.from_pretrained(spec.model_dir, local_files_only=True), spec.model_dir)
        _MODEL.extend((DS.load_model(spec.model_dir), tok, spec))
        if not (INTERP/'vocab.json').exists():
            write_vocab(tok)
    return _MODEL


def load_stage(name, collect):
    """artifacts/interp/<name>.json, collected with collect(model, tokenizer, spec) if missing."""
    path = INTERP/f'{name}.json'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'model': 'deepseek_v4_flash', **collect(*model())}, ensure_ascii=False))
    return json.loads(path.read_text())


def finals(run):
    return [p for p in json.loads((run/'agent_prompts.json').read_text()) if p['phase'] == 'final']


def write_vocab(tok):
    """Display text of every token id, and its raw bytes (hex) so generated text can be rebuilt exactly."""
    # GPT-2 byte-level alphabet (DeepSeek's tokenizer is byte-level BPE): printable bytes map to themselves,
    # the rest to chr(256 + n).
    keep = [*range(ord('!'), ord('~') + 1), *range(ord('¡'), ord('¬') + 1), *range(ord('®'), ord('ÿ') + 1)]
    extra = [b for b in range(256) if b not in keep]
    byte_of = {**{chr(b): b for b in keep}, **{chr(256 + n): b for n, b in enumerate(extra)}}
    raw = lambda t: bytes(byte_of[c] for c in t) if all(c in byte_of for c in t) else t.encode()
    ids = range(len(tok))
    (INTERP/'vocab.json').write_text(json.dumps({'text': [x.strip() for x in tok.batch_decode([[i] for i in ids])],
                                                 'bytes': [raw(t).hex() for t in tok.convert_ids_to_tokens(list(ids))]},
                                                ensure_ascii=False))


def vocab():
    if not _VOCAB:
        v = json.loads((INTERP/'vocab.json').read_text())
        _VOCAB.update(text=v['text'], low=[t.lower() for t in v['text']], bytes=v['bytes'])
    return _VOCAB
