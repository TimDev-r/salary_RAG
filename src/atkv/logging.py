"""Structured JSON logging.

Every query gets a trace_id, and the retrieved chunk ids and their component
scores are logged with it. That is what makes an answer REPLAYABLE: when a
figure comes out wrong, the log says which chunks were retrieved, which
retriever surfaced each one, and what the model was shown -- without
re-running anything and without hoping the bug reproduces.

Component scores are kept apart (dense, lexical, fused, rerank) rather than
collapsed into one number, because the useful question is almost always "which
retriever dragged this in", and a single fused score cannot answer it.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# Libraries that log every HTTP call at INFO. Left alone, a single model load
# emits ~30 lines about huggingface.co and buries the one line that matters.
# A log nobody can read is not an audit trail.
NOISY = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock",
         "sentence_transformers", "transformers")


def configure(level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(level)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
