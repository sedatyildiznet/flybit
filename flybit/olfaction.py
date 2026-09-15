"""Synthetic desktop odor field for Flybit's feeding environment.

The field is a modeled world signal. It is deliberately not injected into
MaleCNS until receptor-level olfactory identities are available and validated
in the bundled metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OdorSnapshot:
    left: float
    right: float
    mean: float
    gradient: float
    salience: float
    food_distance: float | None


class FoodOdorModel:
    """Sample a smooth sugar-associated odor field at two virtual antennae."""

    def __init__(
        self,
        *,
        diffusion_length: float = 180.0,
        antenna_forward: float = 15.0,
        antenna_lateral: float = 6.0,
    ) -> None:
        self.diffusion_length = max(20.0, float(diffusion_length))
        self.antenna_forward = float(antenna_forward)
        self.antenna_lateral = float(antenna_lateral)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _concentration(
        self,
        x: float,
        y: float,
        *,
        source_x: float,
        source_y: float,
        amount: float,
    ) -> float:
        distance = math.hypot(x - source_x, y - source_y)
        smooth = math.exp(-distance / self.diffusion_length)
        near = 1.0 / (1.0 + (distance / 110.0) ** 2)
        return self._clamp01(max(0.1, amount) * smooth * near)

    def sample(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        food: tuple[float, float, float] | None,
        hunger_drive: float,
    ) -> OdorSnapshot:
        if food is None:
            return OdorSnapshot(
                left=0.0,
                right=0.0,
                mean=0.0,
                gradient=0.0,
                salience=0.0,
                food_distance=None,
            )

        source_x, source_y, amount = food
        cs = math.cos(heading)
        sn = math.sin(heading)

        forward_x = x + cs * self.antenna_forward
        forward_y = y + sn * self.antenna_forward
        lateral_x = -sn * self.antenna_lateral
        lateral_y = cs * self.antenna_lateral

        left = self._concentration(
            forward_x + lateral_x,
            forward_y + lateral_y,
            source_x=source_x,
            source_y=source_y,
            amount=amount,
        )
        right = self._concentration(
            forward_x - lateral_x,
            forward_y - lateral_y,
            source_x=source_x,
            source_y=source_y,
            amount=amount,
        )
        mean = 0.5 * (left + right)
        gradient = (
            (right - left) / max(left + right, 1e-6)
            if mean > 0.0
            else 0.0
        )
        hunger = self._clamp01(hunger_drive)
        salience = self._clamp01(
            mean * (0.25 + 0.75 * hunger)
        )

        return OdorSnapshot(
            left=left,
            right=right,
            mean=mean,
            gradient=max(-1.0, min(1.0, gradient)),
            salience=salience,
            food_distance=math.hypot(
                float(source_x) - float(x),
                float(source_y) - float(y),
            ),
        )
