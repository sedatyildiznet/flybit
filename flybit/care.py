"""Care/feeding state that does not choose neural behaviour."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

from .state import FlybitState


@dataclass
class FoodDrop:
    x: float
    y: float
    amount: float = 1.0
    radius: float = 15.0


class CareModel:
    """Simple persistent nutritional state.

    Hunger is app-level physiology, not a movement controller. It never selects
    an action or injects locomotor commands into MaleCNS.
    """

    # Roughly four hours of continuous runtime from recently-fed to fully hungry.
    HUNGER_SECONDS = 4.0 * 60.0 * 60.0

    def __init__(self, state: FlybitState) -> None:
        self.state = state
        self.food: FoodDrop | None = None

    def tick(self, dt: float) -> None:
        self.state.hunger = min(
            1.0,
            max(
                0.0,
                self.state.hunger
                + max(0.0, float(dt)) / self.HUNGER_SECONDS,
            ),
        )

    def place_food(
        self,
        x: float,
        y: float,
        *,
        amount: float = 1.0,
    ) -> FoodDrop:
        self.food = FoodDrop(
            x=float(x),
            y=float(y),
            amount=max(0.1, float(amount)),
        )
        return self.food

    def contact(
        self,
        x: float,
        y: float,
        *,
        body_radius: float = 14.0,
    ) -> bool:
        food = self.food
        if food is None:
            return False

        distance = math.hypot(
            float(x) - food.x,
            float(y) - food.y,
        )
        if distance > food.radius + body_radius:
            return False

        reduction = min(
            0.55,
            0.32 * food.amount,
        )
        self.state.hunger = max(
            0.0,
            self.state.hunger - reduction,
        )
        self.state.feedings += 1
        self.state.last_feed_at = datetime.now(
            timezone.utc
        ).isoformat()
        self.food = None
        return True
