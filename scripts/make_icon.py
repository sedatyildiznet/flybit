"""Generate Flybit.ico for PyInstaller using Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def make(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle(
        (8, 8, size - 8, size - 8),
        radius=52,
        fill=(13, 17, 23, 255),
    )
    d.ellipse(
        (48, 48, 208, 208),
        outline=(93, 216, 242, 78),
        width=12,
    )

    # Wings.
    d.ellipse((48, 58, 130, 111), fill=(170, 214, 220, 72), outline=(170, 214, 220, 150), width=4)
    d.ellipse((58, 142, 140, 195), fill=(170, 214, 220, 72), outline=(170, 214, 220, 150), width=4)

    # Legs.
    for xy in (
        (92, 112, 56, 76),
        (112, 106, 110, 62),
        (132, 112, 174, 78),
        (92, 144, 56, 180),
        (112, 150, 110, 194),
        (132, 144, 174, 178),
    ):
        d.line(xy, fill=(220, 225, 229, 190), width=5)

    # Body.
    d.ellipse((76, 105, 148, 151), fill=(217, 225, 230, 255))
    d.ellipse((130, 100, 184, 156), fill=(104, 116, 128, 255))
    d.ellipse((170, 106, 212, 150), fill=(72, 82, 92, 255))
    d.ellipse((194, 112, 209, 127), fill=(230, 87, 70, 255))
    d.ellipse((194, 130, 209, 145), fill=(230, 87, 70, 255))
    d.ellipse((119, 123, 129, 133), fill=(93, 216, 242, 255))
    return img


if __name__ == "__main__":
    out = Path("build")
    out.mkdir(exist_ok=True)
    icon = make()
    icon.save(
        out / "flybit.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(out / "flybit.ico")
