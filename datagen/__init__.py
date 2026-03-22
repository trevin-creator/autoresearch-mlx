"""datagen — Stereo event camera data generation for autoresearch-mlx."""

from .config import CameraConfig, EventConfig, OutputConfig, SimConfig, StereoConfig
from .emulator import EventCameraEmulator
from .trajectory import ScriptedRigTrajectory
from .writer import DatasetWriter

__all__ = [
    "CameraConfig",
    "DatasetWriter",
    "EventCameraEmulator",
    "EventConfig",
    "OutputConfig",
    "ScriptedRigTrajectory",
    "SimConfig",
    "StereoConfig",
]
