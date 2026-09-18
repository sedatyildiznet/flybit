"""Six-leg modified-tripod gait and appendage pose model for Flybit."""
from __future__ import annotations

from dataclasses import dataclass
import math


LEG_NAMES = ("LF", "LM", "LH", "RF", "RM", "RH")
TRIPOD_A = {"LF", "RM", "LH"}
TRIPOD_B = {"RF", "LM", "RH"}


@dataclass(frozen=True)
class LegPose:
    name: str
    root_x: float
    root_y: float
    foot_x: float
    foot_y: float
    lift: float
    stance: bool


@dataclass(frozen=True)
class GaitSnapshot:
    phase: float
    stride_hz: float
    stance_count: int
    tripod_coherence: float
    support_factor: float
    leg_extension: float
    body_bob: float
    legs: tuple[LegPose, ...]


class SixLegGait:
    """Compact kinematic approximation of adult Drosophila leg coordination.

    Walking uses two alternating modified-tripod groups. The model is not a
    musculoskeletal solver; it provides explicit stance/swing contacts and a
    support term that can couple back into body motion.
    """

    _ROOTS = {
        "LF": (4.7, -2.8),
        "LM": (0.3, -3.8),
        "LH": (-4.7, -2.9),
        "RF": (4.7, 2.8),
        "RM": (0.3, 3.8),
        "RH": (-4.7, 2.9),
    }
    _REST_FEET = {
        "LF": (9.5, -8.0),
        "LM": (0.6, -9.6),
        "LH": (-9.2, -7.7),
        "RF": (9.5, 8.0),
        "RM": (0.6, 9.6),
        "RH": (-9.2, 7.7),
    }

    def __init__(self, *, stride_scale: float = 1.0) -> None:
        self.phase = 0.0
        self.stride_scale = max(0.8, min(1.2, float(stride_scale)))
        self._snapshot = self._static_snapshot()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _static_snapshot(self) -> GaitSnapshot:
        legs = tuple(
            LegPose(
                name=name,
                root_x=self._ROOTS[name][0],
                root_y=self._ROOTS[name][1],
                foot_x=self._REST_FEET[name][0],
                foot_y=self._REST_FEET[name][1],
                lift=0.0,
                stance=True,
            )
            for name in LEG_NAMES
        )
        return GaitSnapshot(
            phase=self.phase,
            stride_hz=0.0,
            stance_count=6,
            tripod_coherence=1.0,
            support_factor=1.0,
            leg_extension=0.0,
            body_bob=0.0,
            legs=legs,
        )

    @property
    def snapshot(self) -> GaitSnapshot:
        return self._snapshot

    def update(
        self,
        dt: float,
        *,
        speed_norm: float,
        turn_rate: float = 0.0,
        airborne: bool = False,
        landing_drive: float = 0.0,
        takeoff_preload: float = 0.0,
        groom_target: str = "",
        micro_action: str = "",
    ) -> GaitSnapshot:
        dt = max(0.001, min(0.20, float(dt)))
        speed = self._clamp01(speed_norm)
        landing = self._clamp01(landing_drive)
        preload = self._clamp01(takeoff_preload)

        stride_hz = 0.0 if airborne else (1.1 + 8.9 * speed)
        if speed < 0.025:
            stride_hz = 0.0
        self.phase = (self.phase + dt * stride_hz) % 1.0

        if airborne:
            extension = max(0.0, min(1.0, landing))
            legs = []
            for name in LEG_NAMES:
                rx, ry = self._ROOTS[name]
                fx, fy = self._REST_FEET[name]
                # In free flight the legs stay compact. As landing drive
                # rises, all six legs extend toward their grounded contact
                # geometry so visual pose and physical landing state agree.
                reach = 0.26 + 0.74 * extension
                legs.append(
                    LegPose(
                        name=name,
                        root_x=rx,
                        root_y=ry,
                        foot_x=rx + (fx - rx) * reach,
                        foot_y=ry + (fy - ry) * reach,
                        lift=1.0 - extension,
                        stance=False,
                    )
                )
            self._snapshot = GaitSnapshot(
                phase=self.phase,
                stride_hz=0.0,
                stance_count=0,
                tripod_coherence=0.0,
                support_factor=0.0,
                leg_extension=extension,
                body_bob=0.0,
                legs=tuple(legs),
            )
            return self._snapshot

        if speed < 0.025 and not groom_target and preload <= 0.0:
            snap = self._static_snapshot()
            if micro_action == "leg_adjust":
                poses = list(snap.legs)
                pose = poses[1]
                poses[1] = LegPose(
                    pose.name,
                    pose.root_x,
                    pose.root_y,
                    pose.foot_x + 1.8,
                    pose.foot_y - 0.8,
                    0.5,
                    False,
                )
                snap = GaitSnapshot(
                    phase=snap.phase,
                    stride_hz=0.0,
                    stance_count=5,
                    tripod_coherence=0.92,
                    support_factor=0.98,
                    leg_extension=0.0,
                    body_bob=0.0,
                    legs=tuple(poses),
                )
            self._snapshot = snap
            return snap

        stance_fraction = max(0.52, min(0.72, 0.70 - 0.17 * speed))
        stride = (2.4 + 4.8 * speed) * self.stride_scale
        turn = max(-1.0, min(1.0, float(turn_rate) / 3.2))
        poses: list[LegPose] = []
        a_stance = 0
        b_stance = 0

        for name in LEG_NAMES:
            offset = 0.0 if name in TRIPOD_A else 0.5
            p = (self.phase + offset) % 1.0
            rx, ry = self._ROOTS[name]
            base_x, base_y = self._REST_FEET[name]

            if speed < 0.025:
                stance = True
                lift = 0.0
                fx, fy = base_x, base_y
            elif p < stance_fraction:
                stance = True
                q = p / stance_fraction
                fx = base_x + stride * (0.52 - q)
                fy = base_y
                lift = 0.0
            else:
                stance = False
                q = (p - stance_fraction) / max(1e-6, 1.0 - stance_fraction)
                fx = base_x + stride * (-0.48 + q)
                fy = base_y
                lift = math.sin(math.pi * q)

            side = -1.0 if name.startswith("L") else 1.0
            fx *= 1.0 - side * turn * 0.07
            fy *= 1.0 + side * turn * 0.10

            if name in TRIPOD_A and stance:
                a_stance += 1
            if name in TRIPOD_B and stance:
                b_stance += 1

            poses.append(
                LegPose(
                    name=name,
                    root_x=rx,
                    root_y=ry,
                    foot_x=fx,
                    foot_y=fy,
                    lift=lift,
                    stance=stance,
                )
            )

        if groom_target:
            phase = math.sin(self.phase * math.tau * 2.0)
            pose_map = {p.name: p for p in poses}
            if groom_target in {"eyes", "antennae", "proboscis"}:
                for name, ysign in (("LF", -1.0), ("RF", 1.0)):
                    p = pose_map[name]
                    pose_map[name] = LegPose(
                        p.name,
                        p.root_x,
                        p.root_y,
                        8.5 + 1.2 * abs(phase),
                        ysign * (1.2 + 1.4 * phase),
                        0.8,
                        False,
                    )
            elif groom_target in {"abdomen", "wings", "thorax"}:
                for name, ysign in (("LH", -1.0), ("RH", 1.0)):
                    p = pose_map[name]
                    tx = -5.8 if groom_target == "abdomen" else (-1.5 if groom_target == "wings" else 1.5)
                    pose_map[name] = LegPose(
                        p.name,
                        p.root_x,
                        p.root_y,
                        tx + 1.5 * phase,
                        ysign * (2.2 + 1.1 * abs(phase)),
                        0.85,
                        False,
                    )
            poses = [pose_map[name] for name in LEG_NAMES]

        stance_count = sum(1 for p in poses if p.stance)
        dominant_tripod = max(a_stance, b_stance)
        coherence = dominant_tripod / 3.0 if speed >= 0.025 else 1.0
        support = max(0.72, min(1.05, 0.82 + 0.22 * coherence))
        if preload > 0.0:
            support = min(1.12, support + 0.10 * preload)

        body_bob = (
            math.sin(self.phase * math.tau * 2.0)
            * 0.55
            * speed
            * (1.0 - 0.55 * preload)
        )

        self._snapshot = GaitSnapshot(
            phase=self.phase,
            stride_hz=stride_hz,
            stance_count=stance_count,
            tripod_coherence=coherence,
            support_factor=support,
            leg_extension=0.0,
            body_bob=body_bob,
            legs=tuple(poses),
        )
        return self._snapshot
