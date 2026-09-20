"""Modeled six-leg contact and load sensing for the Flybit body."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .gait import LEG_NAMES, LegPose
from .world import LocalGeometry, Surface, query_local_geometry, support_at


@dataclass(frozen=True)
class LegContact:
    name: str
    contact: bool
    touchdown: bool
    liftoff: bool
    load: float
    slip: float
    support_id: int | None
    edge_distance: float


@dataclass(frozen=True)
class ProprioceptionSnapshot:
    legs: tuple[LegContact, ...]
    contact_count: int
    total_load: float
    support_polygon_area: float
    stability: float
    slip_mean: float
    stumble: bool
    touchdown_count: int
    liftoff_count: int
    geometry: LocalGeometry

    @property
    def landing_contact(self) -> bool:
        # Touchdown may occur one integration step before altitude reaches the
        # body-contact plane, so sustained loaded contact is also valid.
        return self.contact_count >= 3 and self.total_load >= 0.45 and self.stability >= 0.22


class ProprioceptionModel:
    """Convert gait pose and local substrate geometry into contact feedback."""

    def __init__(self) -> None:
        self._contact = {name: True for name in LEG_NAMES}
        self._last_foot: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _world_foot(x: float, y: float, heading: float, leg: LegPose) -> tuple[float, float]:
        cs, sn = math.cos(heading), math.sin(heading)
        return x + leg.foot_x * cs - leg.foot_y * sn, y + leg.foot_x * sn + leg.foot_y * cs

    def update(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        altitude: float,
        vertical_velocity: float,
        legs: tuple[LegPose, ...],
        surfaces: list[Surface],
        bounds: tuple[float, float, float, float],
        dt: float,
        body_speed: float = 0.0,
    ) -> ProprioceptionSnapshot:
        dt = max(0.001, float(dt))
        geometry = query_local_geometry(surfaces, x, y, 32.0, bounds)
        contacts: list[LegContact] = []
        contact_points: list[tuple[float, float]] = []
        stance_candidates = [leg for leg in legs if leg.stance or leg.lift < 0.42]
        load_each = 1.0 / max(1, len(stance_candidates))
        for leg in legs:
            fx, fy = self._world_foot(x, y, heading, leg)
            support = support_at(surfaces, fx, fy)
            support_id = support.id if support is not None else 0
            local = query_local_geometry(surfaces, fx, fy, 24.0, bounds)
            extended_contact = altitude <= max(0.7, 2.3 * (1.0 - leg.lift))
            contact = bool((leg.stance or leg.lift < 0.42) and extended_contact)
            previous = self._contact.get(leg.name, False)
            prior_pos = self._last_foot.get(leg.name, (fx, fy))
            foot_speed = math.hypot(fx - prior_pos[0], fy - prior_pos[1]) / dt
            relative_foot_speed = max(0.0, foot_speed - max(0.0, body_speed))
            slip = 0.0
            if contact:
                slip = min(1.0, max(0.0, (relative_foot_speed - 8.0) / max(25.0, body_speed + 25.0)))
                if local.support_continuity < 0.5:
                    slip = min(1.0, slip + 0.25)
                contact_points.append((fx, fy))
            contacts.append(LegContact(leg.name, contact, contact and not previous, previous and not contact, load_each * (1.0 - 0.5 * slip) if contact else 0.0, slip, support_id if contact else None, local.edge_distance))
            self._contact[leg.name] = contact
            self._last_foot[leg.name] = (fx, fy)

        area = 0.0
        if len(contact_points) >= 3:
            cx = sum(p[0] for p in contact_points) / len(contact_points)
            cy = sum(p[1] for p in contact_points) / len(contact_points)
            ordered = sorted(contact_points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            area = abs(sum(ordered[i][0] * ordered[(i + 1) % len(ordered)][1] - ordered[(i + 1) % len(ordered)][0] * ordered[i][1] for i in range(len(ordered)))) * 0.5
        total_load = sum(item.load for item in contacts)
        slip_mean = sum(item.slip for item in contacts) / max(1, len(contacts))
        stability = min(1.0, (len(contact_points) / 3.0) * min(1.0, area / 120.0) * (1.0 - 0.65 * slip_mean))
        stumble = altitude <= 0.7 and (len(contact_points) < 2 or slip_mean > 0.55 or (vertical_velocity < -35.0 and stability < 0.35))
        return ProprioceptionSnapshot(tuple(contacts), len(contact_points), total_load, area, stability, slip_mean, stumble, sum(i.touchdown for i in contacts), sum(i.liftoff for i in contacts), geometry)
