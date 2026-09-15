from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from .ui import FlybitWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Flybit")
    window = FlybitWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
