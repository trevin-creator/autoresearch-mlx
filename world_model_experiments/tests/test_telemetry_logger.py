from __future__ import annotations

import json

from world_model_experiments.telemetry import TelemetryLogger


def test_telemetry_logger_context_manager_closes_file(tmp_path) -> None:
    out = tmp_path / "telemetry.jsonl"
    with TelemetryLogger(str(out)) as logger:
        logger.log({"kind": "event", "value": 3})
        assert not logger._fh.closed

    assert logger._fh.closed
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["kind"] == "event"
    assert payload["value"] == 3


def test_telemetry_logger_close_is_idempotent(tmp_path) -> None:
    out = tmp_path / "telemetry.jsonl"
    logger = TelemetryLogger(str(out))
    logger.log({"kind": "event", "value": 1})
    logger.close()
    logger.close()
    assert logger._fh.closed
