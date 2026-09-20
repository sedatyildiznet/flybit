"""Bounded sensory-context associative memory.

Fingerprints contain retinal, odor and local-geometry quantities only.  The
resulting biases are deliberately weak inputs to ethology; this module never
issues movement commands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time


MEMORY_SCHEMA = 2


@dataclass(frozen=True)
class SensoryFingerprint:
    retinal_left: float
    retinal_right: float
    optic_flow: float
    odor_left: float
    odor_right: float
    edge_proximity: float
    support_continuity: float

    def vector(self) -> tuple[float, ...]:
        return tuple(float(value) for value in asdict(self).values())


@dataclass
class Association:
    fingerprint: list[float]
    food: float = 0.0
    threat: float = 0.0
    neutral: float = 0.0
    safe_rest: float = 0.0
    eligibility: float = 0.0
    updated_at: float = 0.0
    exposures: int = 0


@dataclass(frozen=True)
class MemoryBias:
    food_bias: float = 0.0
    threat_bias: float = 0.0
    rest_bias: float = 0.0
    attention_bias: float = 0.0
    similarity: float = 0.0


def sensory_fingerprint(*, sensory=None, odor=None, geometry=None) -> SensoryFingerprint:
    return SensoryFingerprint(
        retinal_left=float(getattr(sensory, "retinal_loom_left", 0.0)),
        retinal_right=float(getattr(sensory, "retinal_loom_right", 0.0)),
        optic_flow=max(-1.0, min(1.0, float(getattr(sensory, "optic_flow", 0.0)))),
        odor_left=float(getattr(odor, "left", 0.0)), odor_right=float(getattr(odor, "right", 0.0)),
        edge_proximity=max(0.0, 1.0 - float(getattr(geometry, "edge_distance", 32.0)) / 32.0),
        support_continuity=float(getattr(geometry, "support_continuity", 1.0)),
    )


class AssociativeMemory:
    MAX_ASSOCIATIONS = 128
    DECAY_SECONDS = 14.0 * 86400.0

    def __init__(self, path: Path | None = None, *, max_associations: int = MAX_ASSOCIATIONS) -> None:
        self.path = path
        self.max_associations = max(8, int(max_associations))
        self.associations: list[Association] = []
        if path is not None:
            self.load()

    @staticmethod
    def _similarity(a: tuple[float, ...] | list[float], b: tuple[float, ...] | list[float]) -> float:
        distance = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
        return math.exp(-distance / 0.72)

    def recall(self, fingerprint: SensoryFingerprint, *, now: float | None = None) -> MemoryBias:
        now = time.time() if now is None else float(now)
        ranked: list[tuple[float, Association, float]] = []
        for item in self.associations:
            sim = self._similarity(fingerprint.vector(), item.fingerprint)
            decay = math.exp(-max(0.0, now - item.updated_at) / self.DECAY_SECONDS)
            ranked.append((sim * decay, item, decay))
        ranked.sort(key=lambda row: row[0], reverse=True)
        ranked = ranked[:5]
        total = sum(weight for weight, _, _ in ranked)
        if total <= 1e-8:
            return MemoryBias()
        food = sum(w * a.food * d for w, a, d in ranked) / total
        threat = sum(w * a.threat * d for w, a, d in ranked) / total
        rest = sum(w * a.safe_rest * d for w, a, d in ranked) / total
        neutral = sum(w * a.neutral * d for w, a, d in ranked) / total
        return MemoryBias(max(-.20, min(.20, .16 * food)), max(0.0, min(.30, .22 * threat)),
                          max(-.15, min(.15, .12 * rest)), max(-.15, min(.15, .12 * (food + threat - neutral))), ranked[0][0])

    def observe(self, fingerprint: SensoryFingerprint, *, food_reward: float = 0.0,
                threat: float = 0.0, neutral: float = 0.0, safe_rest: float = 0.0,
                dt: float = .02, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        vector = fingerprint.vector()
        best = max(self.associations, key=lambda item: self._similarity(vector, item.fingerprint), default=None)
        similarity = self._similarity(vector, best.fingerprint) if best else 0.0
        if best is None or similarity < .82:
            best = Association(list(vector), updated_at=now)
            self.associations.append(best)
        trace_decay = math.exp(-max(0.0, dt) / 4.0)
        best.eligibility = min(1.0, best.eligibility * trace_decay + max(0.02, 1.0 - similarity))
        rate = .18 * best.eligibility
        best.food += rate * (max(-1.0, min(1.0, food_reward)) - best.food)
        best.threat += rate * (max(0.0, min(1.0, threat)) - best.threat)
        best.neutral += rate * (max(0.0, min(1.0, neutral)) - best.neutral)
        best.safe_rest += rate * (max(0.0, min(1.0, safe_rest)) - best.safe_rest)
        best.fingerprint = [old + .08 * (new - old) for old, new in zip(best.fingerprint, vector)]
        best.updated_at, best.exposures = now, best.exposures + 1
        self.associations.sort(key=lambda item: (item.updated_at, item.exposures), reverse=True)
        del self.associations[self.max_associations:]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": MEMORY_SCHEMA, "associations": [asdict(item) for item in self.associations]}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.path)

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            schema = int(raw.get("schema", 1))
            rows = raw.get("associations", raw.get("memories", []))
            loaded = []
            for row in rows[:self.max_associations]:
                if schema < 2:
                    row.setdefault("eligibility", 0.0)
                    row.setdefault("exposures", 1)
                    row.setdefault("neutral", 0.0)
                loaded.append(Association(**{key: row[key] for key in Association.__dataclass_fields__ if key in row}))
            self.associations = loaded
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            corrupt = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                os.replace(self.path, corrupt)
            except OSError:
                pass
            self.associations = []

