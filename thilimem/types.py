"""Core data types for thilimem."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

MemoryType = Literal["fact", "preference", "event", "entity"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """One atomic, durable thing worth remembering."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    type: MemoryType = "fact"
    entities: list[str] = Field(default_factory=list)
    importance: float = 0.5  # 0..1, how much it matters long-term
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: Optional[datetime] = None
    source: Optional[str] = None  # provenance: session/turn id

    def touch(self) -> None:
        self.last_used_at = utcnow()
