"""Pinned checkpoints, released lenses and cache locations per model."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyModel:
    key: str
    model_id: str
    revision: str
    model_dir: Path
    lens_file: Path
    root: Path
    layers: tuple[int, ...]
    all_layers: tuple[int, ...]
    public_sites: Path
    identity: Path
    dataset_key: str = ""
    chat_family: str = ""

    @property
    def cohort(self):
        return self.dataset_key or self.key

    @property
    def family(self):
        return self.chat_family or ("qwen" if self.key == "qwen3_32b" else "gpt_oss")

    def readout(self, layers=None, device='cuda'):
        from utils.release import ReleasedReadout
        chosen=set(self.layers if layers is None else layers)
        if self.key=='gpt_oss_20b':
            return ReleasedReadout(Path.cwd(),chosen,device=device)
        return ReleasedReadout(Path.cwd(),chosen,device=device,model_dir=self.model_dir,
                               lens_file=self.lens_file,model_revision=self.revision,model_id=self.model_id)

    def ontology(self):
        """The ontology words (ontology.json) as this checkpoint's single-token ids."""
        import json
        base = json.loads(Path('ontology.json').read_text())
        if self.key == 'gpt_oss_20b':
            return base
        path = self.root/'ontology.json'
        if path.exists():
            return json.loads(path.read_text())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        for group in ('target_ids_by_word', 'control_ids_by_word'):
            for word in base[group]:
                ids = tokenizer.encode(' '+word, add_special_tokens=False)
                if len(ids) != 1 or tokenizer.decode(ids) != ' '+word:
                    raise ValueError(f'ontology word is not one token: {word!r}')
                base[group][word] = ids[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(base, indent=2)+'\n')
        return base


MODELS = {
    'gpt_oss_20b':StudyModel('gpt_oss_20b','openai/gpt-oss-20b',
        '6cee5e81ee83917806bbde320786a8fb61efebee',Path('artifacts/gpt_oss_20b'),
        Path('artifacts/lens_candidate/gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt'),
        Path('artifacts/study'),tuple(range(10,15)),tuple(range(23)),
        Path('artifacts/study/public_sites'),Path('artifacts/study/identity')),
    'qwen3_32b':StudyModel('qwen3_32b','Qwen/Qwen3-32B',
        '9216db5781bf21249d130ec9da846c4624c16137',Path('artifacts/qwen3_32b'),
        Path('artifacts/lens_candidate/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt'),
        Path('artifacts/qwen_reproduction'),tuple(range(26,31)),tuple(range(63)),
        Path('artifacts/qwen_reproduction/public_sites'),Path('artifacts/qwen_reproduction/identity')),
    'llama3_3_70b':StudyModel('llama3_3_70b','meta-llama/Llama-3.3-70B-Instruct',
        '6f6073b423013f6a7d4d9f39144961bfbfbc386b',Path('artifacts/llama3_3_70b'),
        Path('artifacts/lens_candidate/llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt'),
        Path('artifacts/llama3_3_70b_reproduction'),tuple(range(32,38)),tuple(range(79)),
        Path('artifacts/llama3_3_70b_reproduction/public_sites'),Path('artifacts/llama3_3_70b_reproduction/identity'),
        'qwen3_32b','llama'),
    'deepseek_v4_flash':StudyModel('deepseek_v4_flash','deepseek-ai/DeepSeek-V4-Flash',
        '60d8d70770c6776ff598c94bb586a859a38244f1',Path('artifacts/deepseek_v4_flash'),
        Path('artifacts/lens_candidate/deepseek-v4-flash/jlens/NeelNanda-pile-10k/deepseek-v4-flash_jacobian_lens_cblank.pt'),
        Path('artifacts/deepseek_reproduction'),tuple(range(16,21)),tuple(range(42)),
        Path('artifacts/deepseek_reproduction/public_sites'),Path('artifacts/deepseek_reproduction/identity'),
        'qwen3_32b','deepseek'),
}


def get_model(key):
    try:
        return MODELS[key]
    except KeyError as exc:
        raise ValueError(f'unknown model {key!r}; choose {tuple(MODELS)}') from exc
