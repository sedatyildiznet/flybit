"""2.5-D desktop body bridge with explicit six-leg support and flight phases."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .gait import GaitSnapshot, LegPose, SixLegGait
from .phenotype import IndividualPhenotype
from .proprioception import ProprioceptionModel, ProprioceptionSnapshot
from .world import Surface, support_at


_NEUTRAL_PHENOTYPE = IndividualPhenotype(
    stride_scale=1.0,
    turn_bias=0.0,
    pause_scale=1.0,
    grooming_bias=1.0,
    startle_bias=0.0,
    flight_saccade_scale=1.0,
    micro_activity=1.0,
    body_scale=1.0,
    handedness=0.0,
)


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
    pitch: float = 0.0
    roll: float = 0.0
    yaw_rate: float = 0.0
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    left_wing_drive: float = 0.0
    right_wing_drive: float = 0.0
    support_id: int | None = 0
    support_title: str = "Desktop"


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
    stride_hz: float
    stance_count: int
    tripod_coherence: float
    wingbeat_hz: float
    locomotor_load: float
    altitude: float
    vertical_speed: float
    leg_extension: float
    body_bob: float
    takeoff_preload: float
    landing_drive: float
    support_title: str
    legs: tuple[LegPose, ...]
    pitch: float
    roll: float
    yaw_rate: float
    pitch_rate: float
    roll_rate: float
    left_wing_drive: float
    right_wing_drive: float
    proprioception: ProprioceptionSnapshot


class FlyKinematics:
    """Desktop-plane body with a separate virtual altitude axis."""

    BODY_HALF_HEIGHT = 9.0
    BODY_HALF_WIDTH = 12.0
    TAKEOFF_PRELOAD_SECONDS = 0.045

    def __init__(
        self,
        state: FlyBodyState,
        *,
        phenotype: IndividualPhenotype | None = None,
    ) -> None:
        self.state = state
        self.phenotype = phenotype or _NEUTRAL_PHENOTYPE
        self._forward = 0.0
        self._backward = 0.0
        self._steer = 0.0
        self._escape = 0.0
        self._escape_bias = 0.0
        self._escape_prev = 0.0
        self._wingbeat_hz = 0.0
        self._last_speed = math.hypot(state.vx, state.vy)
        self._acceleration = 0.0
        self._locomotor_load = 0.0
        self._takeoff_countdown = 0.0
        self._takeoff_pending = False
        self._landing_drive = 0.0
        self._gait = SixLegGait(stride_scale=self.phenotype.stride_scale)
        self._gait_snapshot = self._gait.snapshot
        self._proprioception = ProprioceptionModel()
        self._proprio_snapshot: ProprioceptionSnapshot | None = None

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

    def _begin_airborne_escape(self, events: list[MotionEvent]) -> None:
        s = self.state
        s.heading = self._wrap_angle(
            s.heading + self._escape_bias * 0.72
        )
        s.airborne = True
        s.support_id = None
        s.support_title = "Air"
        s.flight_energy = max(
            s.flight_energy,
            0.55 + self._escape * 0.75,
        )
        s.vertical_velocity = max(
            s.vertical_velocity,
            95.0 + 85.0 * self._escape,
        )
        s.altitude = max(s.altitude, 0.5)
        self._takeoff_pending = False
        self._takeoff_countdown = 0.0
        events.append(
            MotionEvent(
                "takeoff",
                "leg preload -> jump -> wing-powered escape",
            )
        )

    def update(
        self,
        motor: MotorActivity,
        surfaces: list[Surface],
        bounds: tuple[float, float, float, float],
        dt: float = 0.020,
        physiology_gain: float = 1.0,
        *,
        landing_drive: float = 0.0,
        groom_target: str = "",
        micro_action: str = "",
        optic_flow: float = 0.0,
    ) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        s = self.state
        dt = max(0.001, min(0.20, float(dt)))
        physiology_gain = max(0.05, min(1.25, float(physiology_gain)))
        self._landing_drive = max(0.0, min(1.0, float(landing_drive)))

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
        self._escape_bias = self._lowpass(
            self._escape_bias,
            motor.escape_right - motor.escape_left,
            dt,
            0.035,
        )

        normal_turn_rate = max(
            -2.8,
            min(2.8, self._steer * 5.0),
        )
        target_turn_rate = max(
            -4.5,
            min(
                4.5,
                normal_turn_rate + self._escape_bias * 4.6,
            ),
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

        # Grounded escape begins with a short leg-compression phase before the
        # body becomes airborne. This preserves low latency while avoiding an
        # instantaneous teleport from walking to flight.
        if (
            not s.airborne
            and not self._takeoff_pending
            and self._escape > 0.06
            and self._escape_prev <= 0.06
        ):
            self._takeoff_pending = True
            self._takeoff_countdown = self.TAKEOFF_PRELOAD_SECONDS
            events.append(
                MotionEvent(
                    "takeoff_prepare",
                    "middle/hind leg preload",
                )
            )

        self._escape_prev = self._escape

        if self._takeoff_pending and not s.airborne:
            self._takeoff_countdown -= dt
            if self._takeoff_countdown <= 0.0:
                self._begin_airborne_escape(events)

        takeoff_preload = (
            max(
                0.0,
                min(
                    1.0,
                    self._takeoff_countdown
                    / self.TAKEOFF_PRELOAD_SECONDS,
                ),
            )
            if self._takeoff_pending
            else 0.0
        )

        if motor.flight > 0.04 and s.airborne:
            s.flight_energy = max(
                s.flight_energy,
                0.18 + 0.55 * motor.flight,
            )

        if s.flight_energy > 0.0:
            burn = 1.0 + 0.55 * self._landing_drive
            s.flight_energy = max(
                0.0,
                s.flight_energy - dt * burn,
            )

        if s.airborne:
            s.left_wing_drive = self._lowpass(s.left_wing_drive, max(motor.flight_left, self._escape), dt, 0.035)
            s.right_wing_drive = self._lowpass(s.right_wing_drive, max(motor.flight_right, self._escape), dt, 0.035)
            wing_sum = 0.5 * (s.left_wing_drive + s.right_wing_drive)
            wing_difference = s.right_wing_drive - s.left_wing_drive
            # Bilateral sum provides lift/thrust; difference produces roll and
            # yaw.  These are modeled body dynamics, not MaleCNS measurements.
            flow = max(-2.0, min(2.0, float(optic_flow)))
            s.roll_rate += (wing_difference * 22.0 - s.roll_rate * 5.5 - flow * 2.4) * dt
            s.yaw_rate += (wing_difference * 13.0 - s.yaw_rate * 4.2 - flow * 1.8) * dt
            pitch_target = max(-0.32, min(0.35, 0.12 - 0.28 * self._landing_drive))
            s.pitch_rate += ((pitch_target - s.pitch) * 12.0 - s.pitch_rate * 5.0) * dt
            s.roll = max(-0.75, min(0.75, s.roll + s.roll_rate * dt))
            s.pitch = max(-0.55, min(0.55, s.pitch + s.pitch_rate * dt))
            s.heading = self._wrap_angle(s.heading + s.yaw_rate * dt)
            gravity = 180.0
            lift = (
                wing_sum * 230.0
                + self._escape * 300.0
            ) * physiology_gain
            lift *= 1.0 - 0.82 * self._landing_drive
            descent = 145.0 * self._landing_drive

            s.vertical_velocity += (
                lift - gravity - descent
            ) * dt
            s.vertical_velocity *= math.exp(
                -(1.35 + 0.55 * self._landing_drive) * dt
            )
            s.altitude += s.vertical_velocity * dt

            if s.altitude >= 110.0:
                s.altitude = 110.0
                s.vertical_velocity = min(0.0, s.vertical_velocity)

            if s.altitude <= 0.0:
                s.altitude = 0.0
        else:
            s.altitude = 0.0
            s.vertical_velocity = 0.0

        walk_drive = (self._forward - self._backward) * physiology_gain
        planned_ground_norm = min(1.0, abs(walk_drive))
        self._gait_snapshot = self._gait.update(
            dt,
            speed_norm=planned_ground_norm,
            turn_rate=s.angular_velocity,
            airborne=s.airborne,
            landing_drive=max(
                self._landing_drive,
                (1.0 - min(1.0, s.altitude / 8.0))
                if s.airborne and self._escape < 0.025 and motor.flight < 0.04
                else 0.0,
            ),
            takeoff_preload=takeoff_preload,
            groom_target=groom_target,
            micro_action=micro_action,
        )
        self._proprio_snapshot = self._proprioception.update(
            x=s.x, y=s.y, heading=s.heading, altitude=s.altitude,
            vertical_velocity=s.vertical_velocity, legs=tuple(self._gait_snapshot.legs),
            surfaces=surfaces, bounds=bounds, dt=dt,
            body_speed=math.hypot(s.vx, s.vy),
        )
        if s.airborne and s.altitude <= 0.0:
            passive = self._escape < 0.025 and motor.flight < 0.04
            if self._proprio_snapshot.landing_contact and (self._landing_drive > 0.18 or passive):
                s.airborne = False
                s.vertical_velocity = 0.0
                s.flight_energy = 0.0
                s.pitch = s.roll = 0.0
                events.append(MotionEvent("land", "proprioceptive multi-leg substrate contact"))
            else:
                # Contact was not stable: abort or recover instead of declaring
                # a landing from altitude alone.
                s.altitude = 0.5
                s.vertical_velocity = max(20.0, s.vertical_velocity)
                if self._proprio_snapshot.stumble:
                    events.append(MotionEvent("landing_abort", "unstable leg contact"))

        if self._proprio_snapshot.stumble and not s.airborne:
            s.vx *= 0.42
            s.vy *= 0.42
            s.angular_velocity += (0.7 if self.phenotype.handedness >= 0.0 else -0.7)
            events.append(MotionEvent("stumble_recovery", "load redistribution"))

        if s.airborne:
            speed_target = (
                walk_drive * 180.0
                + motor.flight * 360.0 * physiology_gain
                + self._escape * 760.0 * physiology_gain
            )
            speed_target *= 1.0 - 0.42 * self._landing_drive
            response_tau = 0.055
        else:
            contact_gain = self._gait_snapshot.support_factor
            speed_target = walk_drive * 155.0 * contact_gain
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

        drag = math.exp(
            -(2.2 if s.airborne else 4.0) * dt
        )
        s.vx *= drag
        s.vy *= drag

        s.x += s.vx * dt
        s.y += s.vy * dt

        speed = math.hypot(s.vx, s.vy)
        self._acceleration = (
            speed - self._last_speed
        ) / max(dt, 1e-6)
        self._last_speed = speed
        self._locomotor_load = max(
            0.0,
            min(
                1.0,
                speed / (520.0 if s.airborne else 170.0),
            ),
        )

        if s.airborne:
            self._wingbeat_hz = 172.0 + 52.0 * max(
                motor.flight,
                self._escape,
            )
        else:
            self._wingbeat_hz = 0.0

        left, top, right, bottom = bounds
        min_x = left + self.BODY_HALF_WIDTH
        max_x = right - self.BODY_HALF_WIDTH
        min_y = top + self.BODY_HALF_HEIGHT
        max_y = bottom - self.BODY_HALF_HEIGHT

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

        if s.airborne and s.altitude > 0.5:
            next_support_id = None
            next_support_title = "Air"
        else:
            support = support_at(surfaces, s.x, s.y)
            next_support_id = support.id if support is not None else 0
            next_support_title = (
                (support.title or support.kind.title())
                if support is not None
                else "Desktop"
            )

        if (
            next_support_id != s.support_id
            or next_support_title != s.support_title
        ):
            previous = s.support_title
            s.support_id = next_support_id
            s.support_title = next_support_title
            if next_support_title != "Air":
                events.append(
                    MotionEvent(
                        "surface_contact",
                        f"{previous} -> {next_support_title}",
                    )
                )

        return events

    def biomechanics(self) -> BiomechanicsSnapshot:
        gait: GaitSnapshot = self._gait_snapshot
        proprio = self._proprio_snapshot
        if proprio is None:
            proprio = self._proprioception.update(
                x=self.state.x, y=self.state.y, heading=self.state.heading,
                altitude=self.state.altitude, vertical_velocity=self.state.vertical_velocity,
                legs=tuple(gait.legs), surfaces=[],
                bounds=(-100000.0, -100000.0, 100000.0, 100000.0), dt=0.02,
            )
        return BiomechanicsSnapshot(
            speed=float(self._last_speed),
            acceleration=float(self._acceleration),
            turn_rate=float(self.state.angular_velocity),
            gait_phase=float(gait.phase),
            stride_hz=float(gait.stride_hz),
            stance_count=int(gait.stance_count),
            tripod_coherence=float(gait.tripod_coherence),
            wingbeat_hz=float(self._wingbeat_hz),
            locomotor_load=float(self._locomotor_load),
            altitude=float(self.state.altitude),
            vertical_speed=float(self.state.vertical_velocity),
            leg_extension=float(gait.leg_extension),
            body_bob=float(gait.body_bob),
            takeoff_preload=float(
                max(
                    0.0,
                    min(
                        1.0,
                        self._takeoff_countdown
                        / self.TAKEOFF_PRELOAD_SECONDS,
                    ),
                )
                if self._takeoff_pending
                else 0.0
            ),
            landing_drive=float(self._landing_drive),
            support_title=str(self.state.support_title),
            legs=tuple(gait.legs),
            pitch=float(self.state.pitch), roll=float(self.state.roll),
            yaw_rate=float(self.state.yaw_rate), pitch_rate=float(self.state.pitch_rate),
            roll_rate=float(self.state.roll_rate), left_wing_drive=float(self.state.left_wing_drive),
            right_wing_drive=float(self.state.right_wing_drive), proprioception=proprio,
        )
