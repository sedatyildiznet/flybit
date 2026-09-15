"""Windows desktop geometry exposed to the Flybit body simulation.

This module does not choose behaviour. It converts visible top-level windows
into substrate rectangles so body contact/landing can be grounded in the
current desktop geometry. Application names remain observer metadata only.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
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


def support_at(
    surfaces: list[Surface],
    x: float,
    y: float,
) -> Surface | None:
    """Return the topmost visible window under a desktop coordinate.

    A None result means the underlying desktop/glass plane. This is geometry
    only; it never changes heading or chooses a movement.
    """
    candidates = [
        surface
        for surface in surfaces
        if surface.contains(x, y)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda surface: surface.z_order)


class WindowSurfaceScanner:
    """Read visible top-level Windows using the Win32 API.

    Flybit's own process windows are excluded so the organism cannot treat its
    overlay/control panel as an external substrate.
    """

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
            if width < 120 or height < 60:
                return True

            class_name = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(
                hwnd,
                class_name,
                len(class_name),
            )
            if class_name.value in {
                "Progman",
                "WorkerW",
                "Shell_TrayWnd",
                "Shell_SecondaryTrayWnd",
            }:
                return True

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
                )
            )
            return True

        user32.EnumWindows(callback, 0)
        return surfaces
