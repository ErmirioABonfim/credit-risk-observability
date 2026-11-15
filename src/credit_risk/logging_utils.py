"""Logs estruturados (JSON Lines) centralizados para todas as etapas do pipeline.

Cada evento vai para `logs/pipeline.jsonl` e para o console. O formato JSON permite
ingerir os logs em qualquer stack (Loki, Datadog, ELK, CloudWatch) sem parser custom.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

RUN_ID = uuid.uuid4().hex[:12]  # correlaciona todos os eventos de uma execução

_logger = logging.getLogger("credit_risk")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(_h)


def log_event(event: str, level: str = "INFO", **fields: Any) -> dict:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "level": level,
        "event": event,
        **fields,
    }
    with open(config.LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _logger.log(getattr(logging, level), "%s %s", event, json.dumps(fields, ensure_ascii=False, default=str))
    return record


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[dict]:
    """Mede latência e registra sucesso/falha de uma tarefa do pipeline."""
    ctx: dict[str, Any] = {}
    start = time.perf_counter()
    log_event("stage_started", stage=name, **fields)
    try:
        yield ctx
    except Exception as exc:
        ctx["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        log_event("stage_failed", level="ERROR", stage=name, status="failed",
                  error=type(exc).__name__, message=str(exc)[:500], **fields, **ctx)
        raise
    ctx["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
    log_event("stage_succeeded", stage=name, status="success", **fields, **ctx)


def read_events() -> list[dict]:
    if not config.LOG_FILE.exists():
        return []
    with open(config.LOG_FILE, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
