"""Public Qt desktop entrypoint; implementation is split by responsibility."""
from .desktop_controller import FlybitWindow
from .workers import BrainWorker
from .rendering import FlyOverlay, BrainMapWidget, FoodOverlay, FoodPlacementOverlay
from .control_panel import ControlPanel

__all__ = [
    "FlybitWindow", "BrainWorker", "FlyOverlay", "BrainMapWidget",
    "FoodOverlay", "FoodPlacementOverlay", "ControlPanel",
]
