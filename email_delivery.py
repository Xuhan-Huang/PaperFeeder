"""Resolve the digest's delivery time independently of the runner timezone."""

from datetime import datetime, timezone
import re
from typing import Optional
from zoneinfo import ZoneInfo


BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def validate_delivery_settings(mode: str, local_time: str) -> None:
    if mode not in ("immediate", "scheduled"):
        raise ValueError("email_delivery_mode must be immediate or scheduled")
    if not isinstance(local_time, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", local_time):
        raise ValueError("email_delivery_time must be HH:MM in Asia/Shanghai (for example 11:00)")


def scheduled_delivery_time(
    mode: str,
    local_time: str = "11:00",
    *,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Return today's future target in UTC, or None for immediate delivery."""
    validate_delivery_settings(mode, local_time)
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Delivery clock must include a timezone")
    if mode == "immediate":
        return None
    local_now = current.astimezone(BEIJING_TIMEZONE)
    hour, minute = map(int, local_time.split(":"))
    target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target.astimezone(timezone.utc) if target > local_now else None
