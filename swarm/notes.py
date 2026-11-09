"""The shared notes file. Every agent can read new entries at the start of a
turn and append with the write_note tool. Nobody is told what it is for."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, field


@dataclass
class NoteEntry:
    seq: int
    episode_index: int
    phase: str
    author: str
    text: str
    planted: bool = False
    plant_kind: str | None = None
    plant_target: str | None = None


@dataclass
class NotesFile:
    entries: list[NoteEntry] = field(default_factory=list)

    def append(self, episode_index: int, phase: str, author: str, text: str, planted: bool = False,
               plant_kind: str | None = None, plant_target: str | None = None) -> NoteEntry:
        e = NoteEntry(len(self.entries) + 1, episode_index, phase, author, text.strip(), planted, plant_kind, plant_target)
        self.entries.append(e)
        return e

    def new_for(self, agent: str, since_seq: int) -> list[NoteEntry]:
        """Entries this agent has not seen, excluding its own (those are in its context already)."""
        return [e for e in self.entries if e.seq > since_seq and e.author != agent]

    @property
    def last_seq(self) -> int:
        return self.entries[-1].seq if self.entries else 0

    @staticmethod
    def render(entries: list[NoteEntry]) -> str:
        return "\n".join(f"[episode {e.episode_index + 1}] {e.author}: {e.text}" for e in entries)

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.entries]

    def clone(self) -> "NotesFile":
        return copy.deepcopy(self)
