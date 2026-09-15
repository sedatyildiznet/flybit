"""Runtime Flybit application icon drawn with Qt."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)


def flybit_icon(size: int = 256) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Premium dark tile.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(13, 17, 23))
    p.drawRoundedRect(
        QRectF(8, 8, size - 16, size - 16),
        52,
        52,
    )

    # Soft cyan neural halo.
    p.setPen(QPen(QColor(93, 216, 242, 55), 12))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(48, 48, 160, 160))

    p.translate(size / 2, size / 2)
    p.rotate(-12)

    # Wings.
    p.setPen(QPen(QColor(170, 214, 220, 150), 4))
    p.setBrush(QColor(170, 214, 220, 56))
    p.drawEllipse(QRectF(-54, -58, 82, 48))
    p.drawEllipse(QRectF(-44, 10, 82, 48))

    # Legs.
    leg = QPen(QColor(220, 225, 229, 185), 5)
    leg.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(leg)
    for a, b in (
        (QPointF(-28, -18), QPointF(-62, -50)),
        (QPointF(-10, -22), QPointF(-12, -66)),
        (QPointF(10, -18), QPointF(54, -46)),
        (QPointF(-28, 18), QPointF(-62, 50)),
        (QPointF(-10, 22), QPointF(-12, 66)),
        (QPointF(10, 18), QPointF(54, 46)),
    ):
        p.drawLine(a, b)

    # Abdomen / thorax / head.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(217, 225, 230))
    p.drawEllipse(QRectF(-42, -23, 72, 46))
    p.setBrush(QColor(104, 116, 128))
    p.drawEllipse(QRectF(12, -28, 54, 56))
    p.setBrush(QColor(72, 82, 92))
    p.drawEllipse(QRectF(51, -22, 42, 44))

    # Compound eye accent.
    p.setBrush(QColor(230, 87, 70))
    p.drawEllipse(QRectF(72, -16, 15, 15))
    p.drawEllipse(QRectF(72, 1, 15, 15))

    # Neural node accent.
    p.setBrush(QColor(93, 216, 242))
    p.drawEllipse(QRectF(-5, -5, 10, 10))

    p.end()
    return QIcon(pix)
