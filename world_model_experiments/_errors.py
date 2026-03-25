"""Shared error message constants for world_model_experiments (EM101/EM102)."""

from __future__ import annotations

ERR_NO_MOTOR_COMMANDS = "--use-motor-commands set but dataset has no motor_commands key"
ERR_NO_FLIGHT_PLAN = "--use-flight-plan set but dataset has no flight_plan key"
ERR_MOTOR_FP_EXCLUSIVE = "--use-motor-commands and --use-flight-plan are mutually exclusive"
ERR_SEEDS_EMPTY = "At least one seed must be provided"
ERR_SEQ_COUNT_MISMATCH = "features/actions sequence counts do not match"
