"""Semantic desktop perception for Flybit.

This layer labels coarse desktop entities (cursor, windows, native buttons and
well-known applications) for telemetry and sensory context. Labels never map
straight to movement commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import sys


@dataclass(frozen=True)
class PerceivedObject:
    kind: str
    label: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


class DesktopSemanticScanner:
    """Read coarse desktop semantics using Win32 metadata.

    This intentionally avoids OCR and deep vision. It uses OS-level window
    metadata so the organism can distinguish a cursor, a window, a native
    button and common applications such as Chrome.
    """

    APP_HINTS = {
        "chrome": "Chrome",
        "msedge": "Edge",
        "firefox": "Firefox",
        "explorer": "Explorer",
        "code": "VS Code",
        "sublime_text": "Sublime Text",
        "notepad": "Notepad",
        "telegram": "Telegram",
        "discord": "Discord",
        "spotify": "Spotify",
    }

    def __init__(self) -> None:
        self._own_pid = os.getpid()

    def scan(self, cursor_xy: tuple[float, float] | None = None) -> list[PerceivedObject]:
        objects: list[PerceivedObject] = []
        if cursor_xy is not None:
            objects.append(
                PerceivedObject(
                    "cursor", "Mouse cursor",
                    float(cursor_xy[0]) - 8.0,
                    float(cursor_xy[1]) - 8.0,
                    16.0, 16.0, 1.0,
                )
            )
        if sys.platform != "win32":
            return objects

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        def process_name(pid: int) -> str:
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return ""
            try:
                size = wintypes.DWORD(1024)
                buf = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return os.path.splitext(os.path.basename(buf.value))[0].lower()
            finally:
                kernel32.CloseHandle(handle)
            return ""

        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @enum_proc
        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) == self._own_pid:
                return True

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width < 80 or height < 40:
                return True

            title_len = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if title_len > 0:
                buf = ctypes.create_unicode_buffer(min(title_len + 1, 256))
                user32.GetWindowTextW(hwnd, buf, len(buf))
                title = buf.value.strip()

            pname = process_name(int(pid.value))
            app = next((label for key, label in self.APP_HINTS.items() if key in pname), None)
            kind = "application" if app else "window"
            label = app or title or "Window"
            objects.append(
                PerceivedObject(
                    kind, label,
                    float(rect.left), float(rect.top),
                    float(width), float(height),
                    0.98 if app else 0.90,
                )
            )

            # Native Win32 buttons are exposed as child windows with class Button.
            child_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            @child_proc
            def child_callback(child, _child_lparam):
                if not user32.IsWindowVisible(child):
                    return True
                cls = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(child, cls, len(cls))
                if cls.value.lower() != "button":
                    return True
                crect = wintypes.RECT()
                if not user32.GetWindowRect(child, ctypes.byref(crect)):
                    return True
                ctitle_len = user32.GetWindowTextLengthW(child)
                ctitle = "Button"
                if ctitle_len > 0:
                    cbuf = ctypes.create_unicode_buffer(min(ctitle_len + 1, 128))
                    user32.GetWindowTextW(child, cbuf, len(cbuf))
                    ctitle = cbuf.value.strip() or "Button"
                objects.append(
                    PerceivedObject(
                        "button", ctitle,
                        float(crect.left), float(crect.top),
                        float(crect.right - crect.left),
                        float(crect.bottom - crect.top),
                        0.95,
                    )
                )
                return True

            user32.EnumChildWindows(hwnd, child_callback, 0)
            return True

        user32.EnumWindows(callback, 0)
        return objects

    @staticmethod
    def nearest(
        objects: list[PerceivedObject],
        x: float,
        y: float,
        limit: int = 8,
    ) -> tuple[PerceivedObject, ...]:
        ranked = sorted(
            objects,
            key=lambda obj: (obj.center[0] - x) ** 2 + (obj.center[1] - y) ** 2,
        )
        return tuple(ranked[: max(0, int(limit))])
