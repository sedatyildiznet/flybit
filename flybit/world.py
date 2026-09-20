"""Windows desktop geometry exposed to the Flybit body simulation.

The world model remains behaviour-agnostic. It exposes visible substrate
rectangles, support identity and local edge geometry so the ethology layer can
react to boundaries without inspecting application names.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import math
import os
import sys


@dataclass(frozen=True)
class Surface:
    """Visible desktop substrate backed by one top-level native window."""

    id: int
    left: float
    right: float
    top: float
    bottom: float
    title: str = ""
    z_order: int = 0
    kind: str = "window"

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    def contains(
        self,
        x: float,
        y: float,
        *,
        margin: float = 0.0,
    ) -> bool:
        m = max(0.0, float(margin))
        return (
            self.left + m <= float(x) <= self.right - m
            and self.top + m <= float(y) <= self.bottom - m
        )


@dataclass(frozen=True)
class BoundaryCue:
    """Geometric proximity to the current substrate boundary."""

    distance: float
    strength: float
    tangent_heading: float
    inward_heading: float
    edge: str
    source_id: int | None
    source_title: str


@dataclass(frozen=True)
class EdgeSegment:
    """One local, non-semantic substrate boundary segment."""

    surface_id: int | None
    edge: str
    x1: float
    y1: float
    x2: float
    y2: float
    distance: float
    z_order: int


@dataclass(frozen=True)
class LocalGeometry:
    """Geometry surrounding a body point, independent of app metadata."""

    x: float
    y: float
    radius: float
    support_id: int | None
    support_z: int
    edge_distance: float
    corner_distance: float
    overlap_depth: int
    support_continuity: float
    edges: tuple[EdgeSegment, ...]


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _closest_heading(current: float, a: float, b: float) -> float:
    da = abs(_wrap_angle(a - current))
    db = abs(_wrap_angle(b - current))
    return a if da <= db else b


def support_at(
    surfaces: list[Surface],
    x: float,
    y: float,
) -> Surface | None:
    """Return the topmost visible window under a desktop coordinate."""
    candidates = [
        surface
        for surface in surfaces
        if surface.contains(x, y)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda surface: surface.z_order)


def query_local_geometry(
    surfaces: list[Surface],
    x: float,
    y: float,
    radius: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> LocalGeometry:
    """Query nearby edges, overlaps and support continuity.

    Titles are intentionally excluded.  The result is suitable for body and
    proprioceptive mechanics but cannot encode application identity.
    """
    r = max(1.0, float(radius))
    px, py = float(x), float(y)
    support = support_at(surfaces, px, py)
    candidates = list(surfaces)
    if bounds is not None:
        left, top, right, bottom = map(float, bounds)
        candidates.append(Surface(0, left, right, top, bottom, z_order=10**9, kind="desktop"))

    segments: list[EdgeSegment] = []
    corner_distance = float("inf")
    for surface in candidates:
        if px + r < surface.left or px - r > surface.right or py + r < surface.top or py - r > surface.bottom:
            continue
        raw = (
            ("left", surface.left, surface.top, surface.left, surface.bottom),
            ("right", surface.right, surface.top, surface.right, surface.bottom),
            ("top", surface.left, surface.top, surface.right, surface.top),
            ("bottom", surface.left, surface.bottom, surface.right, surface.bottom),
        )
        for edge, x1, y1, x2, y2 in raw:
            nearest_x = min(max(px, min(x1, x2)), max(x1, x2))
            nearest_y = min(max(py, min(y1, y2)), max(y1, y2))
            distance = math.hypot(px - nearest_x, py - nearest_y)
            if distance <= r:
                segments.append(EdgeSegment(surface.id if surface.id else None, edge, x1, y1, x2, y2, distance, surface.z_order))
        for cx, cy in ((surface.left, surface.top), (surface.right, surface.top), (surface.left, surface.bottom), (surface.right, surface.bottom)):
            corner_distance = min(corner_distance, math.hypot(px - cx, py - cy))

    containing = [surface for surface in surfaces if surface.contains(px, py)]
    edge_distance = min((edge.distance for edge in segments), default=r)
    # Probe a small ring; continuity is the fraction retaining the same support.
    probe = min(r, 12.0)
    expected = support.id if support is not None else None
    same = 0
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        found = support_at(surfaces, px + math.cos(angle) * probe, py + math.sin(angle) * probe)
        if (found.id if found is not None else None) == expected:
            same += 1
    return LocalGeometry(
        x=px,
        y=py,
        radius=r,
        support_id=expected,
        support_z=support.z_order if support is not None else 10**9,
        edge_distance=float(edge_distance),
        corner_distance=float(min(corner_distance, r)),
        overlap_depth=len(containing),
        support_continuity=same / 4.0,
        edges=tuple(sorted(segments, key=lambda item: (item.distance, item.z_order))),
    )


def boundary_cue(
    surfaces: list[Surface],
    x: float,
    y: float,
    heading: float,
    bounds: tuple[float, float, float, float],
    *,
    margin: float = 30.0,
) -> BoundaryCue:
    """Return a non-semantic edge cue for the substrate under the body.

    Window borders and desktop/screen borders become 2.5-D environmental
    geometry. Application labels are returned only as telemetry and are never
    required to compute the cue.
    """
    margin = max(1.0, float(margin))
    support = support_at(surfaces, x, y)
    if support is None:
        left, top, right, bottom = map(float, bounds)
        source_id = None
        source_title = "Desktop"
    else:
        left = support.left
        top = support.top
        right = support.right
        bottom = support.bottom
        source_id = support.id
        source_title = support.title or support.kind.title()

    distances = {
        "left": abs(float(x) - left),
        "right": abs(right - float(x)),
        "top": abs(float(y) - top),
        "bottom": abs(bottom - float(y)),
    }
    edge = min(distances, key=distances.get)
    distance = max(0.0, distances[edge])
    strength = max(0.0, min(1.0, (margin - distance) / margin))

    if edge == "left":
        tangent = _closest_heading(heading, math.pi / 2.0, -math.pi / 2.0)
        inward = 0.0
    elif edge == "right":
        tangent = _closest_heading(heading, math.pi / 2.0, -math.pi / 2.0)
        inward = math.pi
    elif edge == "top":
        tangent = _closest_heading(heading, 0.0, math.pi)
        inward = math.pi / 2.0
    else:
        tangent = _closest_heading(heading, 0.0, math.pi)
        inward = -math.pi / 2.0

    return BoundaryCue(
        distance=distance,
        strength=strength,
        tangent_heading=_wrap_angle(tangent),
        inward_heading=_wrap_angle(inward),
        edge=edge,
        source_id=source_id,
        source_title=source_title,
    )


class WindowSurfaceScanner:
    """Read visible top-level Windows using the Win32 API."""

    def __init__(self) -> None:
        self._own_pid = os.getpid()

    def scan(self) -> list[Surface]:
        if sys.platform != "win32":
            return []

        user32 = ctypes.windll.user32
        surfaces: list[Surface] = []
        z_counter = 0

        enum_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        @enum_proc
        def callback(hwnd, _lparam):
            nonlocal z_counter

            current_z = z_counter
            z_counter += 1

            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(
                hwnd,
                ctypes.byref(pid),
            )
            if int(pid.value) == self._own_pid:
                return True

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width < 80 or height < 32:
                return True

            class_name = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(
                hwnd,
                class_name,
                len(class_name),
            )
            cls = class_name.value
            if cls in {"Progman", "WorkerW"}:
                return True

            kind = (
                "taskbar"
                if cls in {"Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
                else "window"
            )

            title_len = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if title_len > 0:
                buf = ctypes.create_unicode_buffer(
                    min(title_len + 1, 256)
                )
                user32.GetWindowTextW(
                    hwnd,
                    buf,
                    len(buf),
                )
                title = buf.value.strip()

            surfaces.append(
                Surface(
                    id=int(hwnd),
                    left=float(rect.left),
                    right=float(rect.right),
                    top=float(rect.top),
                    bottom=float(rect.bottom),
                    title=title,
                    z_order=current_z,
                    kind=kind,
                )
            )
            return True

        user32.EnumWindows(callback, 0)
        return surfaces
