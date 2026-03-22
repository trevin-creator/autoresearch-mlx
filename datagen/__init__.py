"""datagen — Stereo event camera data generation for autoresearch-mlx."""

from .config import CameraConfig, EventConfig, OutputConfig, SimConfig, StereoConfig
from .emulator import EventCameraEmulator
from .optics import SensorImageModel
from .sensor_models import (
    SensorProfile,
    available_sensor_profiles,
    get_sensor_profile,
)
from .trajectory import ScriptedRigTrajectory
from .writer import DatasetWriter

__all__ = [
    "CameraConfig",
    "DatasetWriter",
    "EventCameraEmulator",
    "EventConfig",
    "OutputConfig",
    "ScriptedRigTrajectory",
    "SensorImageModel",
    "SensorProfile",
    "SimConfig",
    "StereoConfig",
    "available_sensor_profiles",
    "get_sensor_profile",
]
