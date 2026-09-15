"""Windows desktop geometry exposed to the Flybit body simulation.

This module does not choose behaviour. It only turns visible top-level Windows
into horizontal physical surfaces that the body can collide with and land on.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import sys


@dataclass(frozen=True)
class Surface:
    """Horizontal top edge of a visible desktop window."""

    id: int
    left: float
    right: float
    top: float
    title: str = ""


class WindowSurfaceScanner:
    """Read visible top-level Windows using the Win32 API.

    Flybit's own process windows are excluded so the fly cannot land on its own
    overlay/control panel.
    """

    def __init__(self) -> None:
        self._own_pid = os.getpid()

    def scan(self) -> list[Surface]:
        if sys.platform != "win32":
            return []

        user32 = ctypes.windll.user32
        surfaces: list[Surface] = []

        enum_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        @enum_proc
        def callback(hwnd, _lparam):
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
                title = buf.value

            surfaces.append(
                Surface(
                    id=int(hwnd),
                    left=float(rect.left),
                    right=float(rect.right),
                    top=float(rect.top),
                    title=title,
                )
            )
            return True

        user32.EnumWindows(callback, 0)
        surfaces.sort(key=lambda s: s.top)
        return surfaces
