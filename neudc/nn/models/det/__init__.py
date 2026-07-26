"""Detector models: the base interface, plain YOLOv8, and SAHI-tiled YOLOv8."""

from .base import BaseDetector
from .sahi import SAHIDetector
from .yolo import YOLOv8

__all__ = ("BaseDetector", "SAHIDetector", "YOLOv8")
