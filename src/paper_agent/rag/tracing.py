"""Human and machine-readable RAG trace sinks."""

import json
from threading import Lock
from typing import TextIO

from paper_agent.rag.domain import RagTraceEvent


class RecordingRagTracer:
    def __init__(self) -> None:
        self.events: list[RagTraceEvent] = []
        self._lock = Lock()

    def emit(self, event: RagTraceEvent) -> None:
        with self._lock:
            self.events.append(event)


class StreamRagTracer:
    def __init__(self, *, mode: str, stream: TextIO) -> None:
        if mode not in {"none", "summary", "jsonl"}:
            raise ValueError("trace mode must be none, summary, or jsonl")
        self._mode = mode
        self._stream = stream
        self._lock = Lock()

    def emit(self, event: RagTraceEvent) -> None:
        if self._mode == "none":
            return
        if self._mode == "jsonl":
            rendered = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        else:
            parts = [event.event]
            if event.round_index is not None:
                parts.append(f"round={event.round_index}")
            if event.work_unit_id is not None:
                parts.append(f"work_unit={event.work_unit_id}")
            parts.extend(f"{key}={value}" for key, value in event.details.items())
            rendered = "[rag] " + " ".join(parts)
        with self._lock:
            self._stream.write(rendered + "\n")
            self._stream.flush()
