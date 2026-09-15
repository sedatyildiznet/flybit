"""Kinematic body bridge driven only by MaleCNS motor read-outs.

The decoder maps identified descending-neuron activity onto simple 2-D forces.
It does not inspect the mouse, windows or application state when choosing an
action. Those belong to the sensory/world side of the closed loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .world import Surface


@dataclass(frozen=True)
class MotorActivity:
    forward_left: float = 0.0
    forward_right: float = 0.0
    steer_left: float = 0.0
    steer_right: float = 0.0
    escape_left: float = 0.0
    escape_right: float = 0.0
    backward_left: float = 0.0
    backward_right: float = 0.0

    @property
    def forward(self) -> float:
        return 0.5 * (self.forward_left + self.forward_right)

    @property
    def backward(self) -> float:
        return 0.5 * (self.backward_left + self.backward_right)

    @property
    def escape(self) -> float:
        return max(self.escape_left, self.escape_right)

    @property
    def steering(self) -> float:
        return self.steer_right - self.steer_left


@dataclass
class FlyBodyState:
    x: float
    y: float
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angular_velocity: float = 0.0
    landed_surface: int | None = None

    @property
    def airborne(self) -> bool:
        return self.landed_surface is None


@dataclass(frozen=True)
class MotionEvent:
    kind: str
    detail: str = ""


class FlyKinematics:
    """Small body model for the Windows overlay.

    This is intentionally not a behaviour state machine. Identified descending
    neurons provide the drive:
      DNg100 -> forward locomotor drive
      DNa02  -> left/right steering differential
      DNp01  -> escape/take-off impulse
      MDN    -> backward locomotor drive

    Gravity, drag and window collision are environment physics.
    """

    BODY_HALF_HEIGHT = 9.0
    BODY_HALF_WIDTH = 12.0

    def __init__(self, state: FlyBodyState) -> None:
        self.state = state
        self._forward = 0.0
        self._backward = 0.0
        self._steer = 0.0
        self._escape = 0.0

    @staticmethod
    def _lowpass(current: float, target: float, dt: float, tau: float) -> float:
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        return current + (target - current) * alpha

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def update(
        self,
        motor: MotorActivity,
        surfaces: list[Surface],
        bounds: tuple[float, float, float, float],
        dt: float = 0.020,
    ) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        s = self.state

        self._forward = self._lowpass(
            self._forward,
            motor.forward,
            dt,
            0.10,
        )
        self._backward = self._lowpass(
            self._backward,
            motor.backward,
            dt,
            0.10,
        )
        self._steer = self._lowpass(
            self._steer,
            motor.steering,
            dt,
            0.08,
        )
        self._escape = self._lowpass(
            self._escape,
            motor.escape,
            dt,
            0.045,
        )

        # DNa02 differential changes orientation. This mapping is a decoder,
        # not an environmental rule.
        s.angular_velocity += self._steer * 8.0 * dt
        s.angular_velocity *= math.exp(-4.0 * dt)
        s.heading = self._wrap_angle(
            s.heading + s.angular_velocity
        )

        surface_by_id = {
            surface.id: surface
            for surface in surfaces
        }

        if s.landed_surface is not None:
            support = surface_by_id.get(s.landed_surface)
            if (
                support is None
                or s.x < support.left - self.BODY_HALF_WIDTH
                or s.x > support.right + self.BODY_HALF_WIDTH
            ):
                s.landed_surface = None
                events.append(MotionEvent("fall", "support lost"))
            else:
                s.y = support.top - self.BODY_HALF_HEIGHT
                direction = 1.0 if math.cos(s.heading) >= 0.0 else -1.0
                walk_drive = self._forward - self._backward
                target_vx = direction * walk_drive * 155.0
                s.vx = self._lowpass(
                    s.vx,
                    target_vx,
                    dt,
                    0.08,
                )
                s.vy = 0.0
                s.x += s.vx * dt

                # DNp01 is the only signal that directly creates a take-off
                # impulse. Cursor distance is never consulted here.
                if self._escape > 0.06:
                    s.landed_surface = None
                    s.vy = -125.0 - 280.0 * self._escape
                    s.vx += math.cos(s.heading) * (
                        220.0 * self._escape
                    )
                    events.append(
                        MotionEvent(
                            "takeoff",
                            "DNp01 escape output",
                        )
                    )
                elif (
                    s.x < support.left
                    or s.x > support.right
                ):
                    s.landed_surface = None
                    events.append(
                        MotionEvent(
                            "fall",
                            "walked beyond window edge",
                        )
                    )

        if s.landed_surface is None:
            previous_y = s.y

            locomotor = self._forward - self._backward
            thrust = locomotor * 185.0 + self._escape * 620.0

            s.vx += math.cos(s.heading) * thrust * dt
            s.vy += math.sin(s.heading) * thrust * dt

            # Escape output also contributes lift, representing the fast
            # take-off/flight command without looking at the stimulus source.
            s.vy -= self._escape * 480.0 * dt

            # Passive body physics.
            s.vy += 145.0 * dt
            drag = math.exp(-1.35 * dt)
            s.vx *= drag
            s.vy *= drag

            s.x += s.vx * dt
            s.y += s.vy * dt

            if s.vy >= 0.0:
                crossed = [
                    surface
                    for surface in surfaces
                    if (
                        surface.left - self.BODY_HALF_WIDTH
                        <= s.x
                        <= surface.right + self.BODY_HALF_WIDTH
                        and previous_y + self.BODY_HALF_HEIGHT
                        <= surface.top
                        <= s.y + self.BODY_HALF_HEIGHT
                    )
                ]
                if crossed:
                    support = min(
                        crossed,
                        key=lambda surface: surface.top,
                    )
                    s.y = support.top - self.BODY_HALF_HEIGHT
                    s.vx *= 0.35
                    s.vy = 0.0
                    s.landed_surface = support.id
                    events.append(
                        MotionEvent(
                            "land",
                            support.title or "window",
                        )
                    )

        left, top, right, bottom = bounds

        # Desktop boundaries are physical containment, not behaviour.
        if s.x < left + self.BODY_HALF_WIDTH:
            s.x = left + self.BODY_HALF_WIDTH
            s.vx = abs(s.vx) * 0.45
            s.heading = self._wrap_angle(math.pi - s.heading)
        elif s.x > right - self.BODY_HALF_WIDTH:
            s.x = right - self.BODY_HALF_WIDTH
            s.vx = -abs(s.vx) * 0.45
            s.heading = self._wrap_angle(math.pi - s.heading)

        if s.y < top + self.BODY_HALF_HEIGHT:
            s.y = top + self.BODY_HALF_HEIGHT
            s.vy = abs(s.vy) * 0.35
        elif s.y > bottom - self.BODY_HALF_HEIGHT:
            s.y = bottom - self.BODY_HALF_HEIGHT
            s.vy = 0.0
            # The desktop bottom acts as a fallback physical surface.
            s.landed_surface = -1
            if not any(event.kind == "land" for event in events):
                events.append(MotionEvent("land", "desktop edge"))

        return events
