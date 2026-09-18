"""Stable individual behavioural/body phenotype for one Flybit organism."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random


@dataclass(frozen=True)
class IndividualPhenotype:
    """Persistent individuality derived deterministically from organism identity.

    These parameters are intentionally small deviations around the species-level
    model. They create stable locomotor fingerprints without allowing a random
    seed to overwhelm neural/sensory behaviour.
    """

    stride_scale: float
    turn_bias: float
    pause_scale: float
    grooming_bias: float
    startle_bias: float
    flight_saccade_scale: float
    micro_activity: float
    body_scale: float
    handedness: float


def phenotype_from_identity(identity: str) -> IndividualPhenotype:
    """Return a stable phenotype for a persistent organism identity."""
    raw = str(identity or "Flybit").encode("utf-8", "replace")
    digest = hashlib.blake2b(raw, digest_size=16, person=b"Flybit-v04").digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = random.Random(seed)

    return IndividualPhenotype(
        stride_scale=rng.uniform(0.92, 1.08),
        turn_bias=rng.uniform(-0.10, 0.10),
        pause_scale=rng.uniform(0.86, 1.18),
        grooming_bias=rng.uniform(0.88, 1.16),
        startle_bias=rng.uniform(-0.055, 0.055),
        flight_saccade_scale=rng.uniform(0.86, 1.18),
        micro_activity=rng.uniform(0.82, 1.18),
        body_scale=rng.uniform(0.94, 1.04),
        handedness=rng.uniform(-1.0, 1.0),
    )
