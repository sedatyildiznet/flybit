from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from .icon import flybit_icon
from .ui import FlybitWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Flybit")
    app.setApplicationDisplayName("Flybit")
    app.setWindowIcon(flybit_icon())
    window = FlybitWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
