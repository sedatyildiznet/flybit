"""2.5-D desktop body bridge driven only by MaleCNS motor read-outs.

Screen x/y remain a flat desktop locomotion plane. A separate virtual altitude
axis models take-off, sustained flight and landing, so gravity never pulls the
organism toward the bottom of the monitor. Identified descending neurons provide
locomotor/steering/escape drive; the body decoder translates those outputs into
planar velocity plus vertical flight dynamics.
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
    altitude: float = 0.0
    vertical_velocity: float = 0.0


@dataclass(frozen=True)
class MotionEvent:
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class BiomechanicsSnapshot:
    speed: float
    acceleration: float
    turn_rate: float
    gait_phase: float
    wingbeat_hz: float
    locomotor_load: float
    altitude: float
    vertical_speed: float


class FlyKinematics:
    """Planar desktop body with an independent virtual altitude axis.

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
        self._gait_phase = 0.0
        self._wingbeat_hz = 0.0
        self._last_speed = math.hypot(state.vx, state.vy)
        self._acceleration = 0.0
        self._locomotor_load = 0.0

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
        physiology_gain: float = 1.0,
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

        # A rising DNp01 response starts a true take-off on a separate virtual
        # altitude axis. This avoids the old mistake of treating screen Y as
        # physical height while still giving flight a real airborne state.
        if (
            self._escape > 0.06
            and self._escape_prev <= 0.06
        ):
            s.airborne = True
            s.flight_energy = max(
                s.flight_energy,
                0.55 + self._escape * 0.75,
            )
            s.vertical_velocity = max(
                s.vertical_velocity,
                95.0 + 85.0 * self._escape,
            )
            s.altitude = max(s.altitude, 0.5)
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

        physiology_gain = max(0.05, min(1.25, float(physiology_gain)))

        if s.airborne:
            gravity = 180.0
            lift = (
                motor.flight * 230.0
                + self._escape * 300.0
            ) * physiology_gain
            s.vertical_velocity += (lift - gravity) * dt
            s.vertical_velocity *= math.exp(-1.35 * dt)
            s.altitude += s.vertical_velocity * dt

            if s.altitude >= 110.0:
                s.altitude = 110.0
                s.vertical_velocity = min(0.0, s.vertical_velocity)

            if s.altitude <= 0.0:
                s.altitude = 0.0
                if (
                    self._escape < 0.025
                    and motor.flight < 0.04
                ):
                    s.airborne = False
                    s.vertical_velocity = 0.0
                    events.append(
                        MotionEvent(
                            "land",
                            "virtual altitude reached desktop plane",
                        )
                    )
                else:
                    s.altitude = 0.5
                    s.vertical_velocity = max(
                        20.0,
                        s.vertical_velocity,
                    )
        else:
            s.altitude = 0.0
            s.vertical_velocity = 0.0


        walk_drive = (self._forward - self._backward) * physiology_gain
        if s.airborne:
            speed_target = (
                walk_drive * 180.0
                + motor.flight * 360.0 * physiology_gain
                + self._escape * 760.0 * physiology_gain
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

        speed = math.hypot(s.vx, s.vy)
        self._acceleration = (speed - self._last_speed) / max(dt, 1e-6)
        self._last_speed = speed
        self._locomotor_load = max(
            0.0,
            min(1.0, speed / (520.0 if s.airborne else 170.0)),
        )
        if s.airborne:
            self._wingbeat_hz = 120.0 + 80.0 * max(
                motor.flight, self._escape
            )
        else:
            self._wingbeat_hz = 0.0
            self._gait_phase = (
                self._gait_phase + dt * (1.5 + 8.5 * self._locomotor_load)
            ) % 1.0

        left, top, right, bottom = bounds
        min_x = left + self.BODY_HALF_WIDTH
        max_x = right - self.BODY_HALF_WIDTH
        min_y = top + self.BODY_HALF_HEIGHT
        max_y = bottom - self.BODY_HALF_HEIGHT

        # Screen edges are physical containment. If the body crosses a wall,
        # reflect only the wall-normal heading component once. This prevents a
        # fly from remaining pinned against a corner while avoiding the old
        # repeated bounce/spin loop.
        hit_left = s.x < min_x
        hit_right = s.x > max_x
        hit_top = s.y < min_y
        hit_bottom = s.y > max_y

        if hit_left:
            s.x = min_x
            if s.vx < 0.0:
                s.vx = abs(s.vx) * 0.28
            s.heading = math.atan2(
                math.sin(s.heading),
                abs(math.cos(s.heading)),
            )
        elif hit_right:
            s.x = max_x
            if s.vx > 0.0:
                s.vx = -abs(s.vx) * 0.28
            s.heading = math.atan2(
                math.sin(s.heading),
                -abs(math.cos(s.heading)),
            )

        if hit_top:
            s.y = min_y
            if s.vy < 0.0:
                s.vy = abs(s.vy) * 0.28
            s.heading = math.atan2(
                abs(math.sin(s.heading)),
                math.cos(s.heading),
            )
        elif hit_bottom:
            s.y = max_y
            if s.vy > 0.0:
                s.vy = -abs(s.vy) * 0.28
            s.heading = math.atan2(
                -abs(math.sin(s.heading)),
                math.cos(s.heading),
            )

        if hit_left or hit_right or hit_top or hit_bottom:
            s.angular_velocity *= 0.25

        return events


    def biomechanics(self) -> BiomechanicsSnapshot:
        """Current body telemetry derived from the physical state."""
        return BiomechanicsSnapshot(
            speed=float(self._last_speed),
            acceleration=float(self._acceleration),
            turn_rate=float(self.state.angular_velocity),
            gait_phase=float(self._gait_phase),
            wingbeat_hz=float(self._wingbeat_hz),
            locomotor_load=float(self._locomotor_load),
            altitude=float(self.state.altitude),
            vertical_speed=float(self.state.vertical_velocity),
        )
