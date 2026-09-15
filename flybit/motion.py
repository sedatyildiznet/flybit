"""2-D desktop body bridge driven only by MaleCNS motor read-outs.

The Windows desktop is treated as a flat locomotion plane, not a vertical world.
There is no downward gravity, falling or edge-bounce rotation. Identified
descending neurons provide locomotor/steering/escape drive; the body decoder
turns those outputs into planar velocity.
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
    flight_left: float = 0.0
    flight_right: float = 0.0

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

    @property
    def flight(self) -> float:
        return 0.5 * (self.flight_left + self.flight_right)


@dataclass
class FlyBodyState:
    x: float
    y: float
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angular_velocity: float = 0.0
    airborne: bool = False
    flight_energy: float = 0.0


@dataclass(frozen=True)
class MotionEvent:
    kind: str
    detail: str = ""


class FlyKinematics:
    """Planar desktop body model.

    Neural mapping:
      DNg100 -> forward locomotor drive
      DNa02  -> steering differential
      DNp01  -> escape/flight burst
      MDN    -> backward locomotor drive
      DNg02  -> flight thrust / wing-power drive

    No cursor/window state is read here. The entire desktop is one flat plane.
    Window contents are visual sensory input, not separate gravity surfaces.
    """

    BODY_HALF_HEIGHT = 9.0
    BODY_HALF_WIDTH = 12.0

    def __init__(self, state: FlyBodyState) -> None:
        self.state = state
        self._forward = 0.0
        self._backward = 0.0
        self._steer = 0.0
        self._escape = 0.0
        self._escape_prev = 0.0

    @staticmethod
    def _lowpass(
        current: float,
        target: float,
        dt: float,
        tau: float,
    ) -> float:
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
        del surfaces  # retained only for API compatibility
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
            0.09,
        )
        self._escape = self._lowpass(
            self._escape,
            motor.escape,
            dt,
            0.045,
        )

        # Steering is a yaw-only decoder on the screen plane. It cannot roll or
        # pitch the rendered body, so the fly no longer appears to somersault.
        target_turn_rate = max(
            -2.8,
            min(2.8, self._steer * 5.0),
        )
        s.angular_velocity = self._lowpass(
            s.angular_velocity,
            target_turn_rate,
            dt,
            0.07,
        )
        s.heading = self._wrap_angle(
            s.heading + s.angular_velocity * dt
        )

        # A rising DNp01 response starts a brief planar flight burst. The body
        # remains on the desktop plane: "airborne" controls wing rendering and
        # motor scale only, not a fake vertical gravity axis.
        if (
            self._escape > 0.06
            and self._escape_prev <= 0.06
        ):
            s.airborne = True
            s.flight_energy = max(
                s.flight_energy,
                0.55 + self._escape * 0.75,
            )
            events.append(
                MotionEvent(
                    "takeoff",
                    "DNp01 escape output",
                )
            )

        self._escape_prev = self._escape

        if motor.flight > 0.04 and s.airborne:
            s.flight_energy = max(
                s.flight_energy,
                0.18 + 0.55 * motor.flight,
            )

        if s.flight_energy > 0.0:
            s.flight_energy = max(
                0.0,
                s.flight_energy - dt,
            )
        elif (
            s.airborne
            and self._escape < 0.025
        ):
            s.airborne = False
            events.append(
                MotionEvent(
                    "land",
                    "desktop plane",
                )
            )

        walk_drive = self._forward - self._backward
        if s.airborne:
            speed_target = (
                walk_drive * 180.0
                + motor.flight * 360.0
                + self._escape * 760.0
            )
            response_tau = 0.055
        else:
            speed_target = walk_drive * 155.0
            response_tau = 0.10

        desired_vx = math.cos(s.heading) * speed_target
        desired_vy = math.sin(s.heading) * speed_target

        s.vx = self._lowpass(
            s.vx,
            desired_vx,
            dt,
            response_tau,
        )
        s.vy = self._lowpass(
            s.vy,
            desired_vy,
            dt,
            response_tau,
        )

        # Passive planar drag prevents endless drifting after neural drive ends.
        drag = math.exp(
            -(2.2 if s.airborne else 4.0) * dt
        )
        s.vx *= drag
        s.vy *= drag

        s.x += s.vx * dt
        s.y += s.vy * dt

        left, top, right, bottom = bounds
        min_x = left + self.BODY_HALF_WIDTH
        max_x = right - self.BODY_HALF_WIDTH
        min_y = top + self.BODY_HALF_HEIGHT
        max_y = bottom - self.BODY_HALF_HEIGHT

        # Screen edges are containment only. We remove the outward velocity
        # component instead of reflecting heading, eliminating edge spin loops.
        if s.x < min_x:
            s.x = min_x
            if s.vx < 0.0:
                s.vx = 0.0
        elif s.x > max_x:
            s.x = max_x
            if s.vx > 0.0:
                s.vx = 0.0

        if s.y < min_y:
            s.y = min_y
            if s.vy < 0.0:
                s.vy = 0.0
        elif s.y > max_y:
            s.y = max_y
            if s.vy > 0.0:
                s.vy = 0.0

        return events
