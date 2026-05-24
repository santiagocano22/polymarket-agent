"""SQLite-backed state: strategy, agent status, and trade history."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    market_id TEXT,
    market_title TEXT,
    token_id TEXT,
    side TEXT,
    size_usdc REAL,
    price REAL,
    status TEXT,
    dry_run INTEGER,
    reasoning TEXT,
    response_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts DESC);
"""

DEFAULTS = {
    "is_active": "false",
    "strategy": "",
    "max_exposure_pct": "30",   # percent of total (USDC + positions value)
    "max_per_trade_usd": "",    # empty = dynamic (10% bankroll)
    "stop_loss_usd": "",        # empty = disabled
    "max_open_positions": "2",  # max 2 posiciones abiertas simultáneas
    "initial_bankroll": "",     # set on first /iniciar
    "loop_interval_seconds": "",   # vacío = usar config
    "trading_hour_start": "",      # vacío = usar config
    "trading_hour_end": "",        # vacío = usar config
    "market_min_volume": "",       # vacío = usar config
    "market_max_volume": "",       # vacío = usar config
    "market_max_days": "",         # vacío = usar config
    "max_trades_day": "6",
    "claude_model": "",            # vacío = usar config
    "daily_tokens_date": "",       # fecha YYYY-MM-DD del contador diario
    "daily_tokens_in": "0",
    "daily_tokens_out": "0",
    "daily_tokens_cached": "0",
}


@dataclass
class Trade:
    id: int
    ts: float
    market_id: str | None
    market_title: str | None
    token_id: str | None
    side: str | None
    size_usdc: float | None
    price: float | None
    status: str | None
    dry_run: bool
    reasoning: str | None


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            for k, v in DEFAULTS.items():
                await db.execute(
                    "INSERT OR IGNORE INTO kv(key, value) VALUES(?, ?)", (k, v)
                )
            await db.commit()

    async def get(self, key: str, default: str = "") -> str:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = await cur.fetchone()
            return row[0] if row else default

    async def set(self, key: str, value: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            await db.commit()

    async def is_active(self) -> bool:
        return (await self.get("is_active", "false")).lower() == "true"

    async def set_active(self, active: bool) -> None:
        await self.set("is_active", "true" if active else "false")

    async def get_strategy(self) -> str:
        return await self.get("strategy", "")

    async def set_strategy(self, strategy: str) -> None:
        await self.set("strategy", strategy)

    async def get_max_exposure_pct(self) -> float:
        return float(await self.get("max_exposure_pct", "50"))

    async def get_max_per_trade_usd(self) -> float | None:
        raw = await self.get("max_per_trade_usd", "")
        return float(raw) if raw else None

    async def get_stop_loss_usd(self) -> float | None:
        raw = await self.get("stop_loss_usd", "")
        return float(raw) if raw else None

    async def get_max_open_positions(self) -> int | None:
        raw = await self.get("max_open_positions", "")
        return int(raw) if raw else None

    async def get_initial_bankroll(self) -> float | None:
        raw = await self.get("initial_bankroll", "")
        return float(raw) if raw else None

    async def set_initial_bankroll(self, value: float) -> None:
        await self.set("initial_bankroll", str(value))

    async def count_trades_today(self) -> int:
        """Count executed (non-dry-run, status=OK) trades since midnight UTC today."""
        import calendar
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        midnight_utc = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        midnight_ts = calendar.timegm(midnight_utc.timetuple())
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """
                SELECT COUNT(*) FROM trades
                WHERE ts >= ?
                  AND dry_run = 0
                  AND status = 'OK'
                """,
                (midnight_ts,),
            )
            row = await cur.fetchone()
        return row[0] if row else 0

    async def last_trade_ts(self) -> float | None:
        """Timestamp of the most recent non-dry-run executed trade today, or None."""
        import calendar
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        midnight_utc = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        midnight_ts = calendar.timegm(midnight_utc.timetuple())
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """
                SELECT ts FROM trades
                WHERE ts >= ?
                  AND dry_run = 0
                ORDER BY ts DESC LIMIT 1
                """,
                (midnight_ts,),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def consecutive_losses_in_category(self, category: str, n: int = 3) -> int:
        """Count consecutive recent losses for trades whose title contains `category`."""
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """
                SELECT status FROM trades
                WHERE LOWER(market_title) LIKE ?
                  AND dry_run = 0
                  AND side = 'BUY'
                ORDER BY ts DESC LIMIT ?
                """,
                (f"%{category.lower()}%", n * 3),
            )
            rows = await cur.fetchall()
        count = 0
        for (status,) in rows:
            if status and status.startswith("FAIL"):
                count += 1
            else:
                break
        return count

    async def log_trade(
        self,
        *,
        market_id: str | None,
        market_title: str | None,
        token_id: str | None,
        side: str | None,
        size_usdc: float | None,
        price: float | None,
        status: str,
        dry_run: bool,
        reasoning: str | None,
        response: dict[str, Any] | None = None,
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO trades
                    (ts, market_id, market_title, token_id, side, size_usdc,
                     price, status, dry_run, reasoning, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    market_id,
                    market_title,
                    token_id,
                    side,
                    size_usdc,
                    price,
                    status,
                    1 if dry_run else 0,
                    reasoning,
                    json.dumps(response) if response else None,
                ),
            )
            await db.commit()

    async def get_loop_interval(self, default: int) -> int:
        raw = await self.get("loop_interval_seconds", "")
        return int(raw) if raw else default

    async def get_trading_hours(self, default_start: int, default_end: int) -> tuple[int, int]:
        s = await self.get("trading_hour_start", "")
        e = await self.get("trading_hour_end", "")
        return (int(s) if s else default_start, int(e) if e else default_end)

    async def get_market_min_volume(self, default: float) -> float:
        raw = await self.get("market_min_volume", "")
        return float(raw) if raw else default

    async def get_market_max_volume(self, default: float) -> float:
        raw = await self.get("market_max_volume", "")
        return float(raw) if raw else default

    async def get_market_max_days(self, default: int) -> int:
        raw = await self.get("market_max_days", "")
        return int(raw) if raw else default

    async def get_max_trades_day(self) -> int:
        raw = await self.get("max_trades_day", "6")
        return int(raw) if raw else 6

    async def get_claude_model(self, default: str) -> str:
        raw = await self.get("claude_model", "")
        return raw if raw else default

    async def accumulate_tokens(self, tokens_in: int, tokens_out: int, tokens_cached: int) -> None:
        """Accumulate daily token usage, resetting counter at UTC midnight."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stored_date = await self.get("daily_tokens_date", "")
        if stored_date != today:
            # Reset for new day
            await self.set("daily_tokens_date", today)
            await self.set("daily_tokens_in", str(tokens_in))
            await self.set("daily_tokens_out", str(tokens_out))
            await self.set("daily_tokens_cached", str(tokens_cached))
        else:
            prev_in     = int(await self.get("daily_tokens_in", "0") or "0")
            prev_out    = int(await self.get("daily_tokens_out", "0") or "0")
            prev_cached = int(await self.get("daily_tokens_cached", "0") or "0")
            await self.set("daily_tokens_in",     str(prev_in     + tokens_in))
            await self.set("daily_tokens_out",    str(prev_out    + tokens_out))
            await self.set("daily_tokens_cached", str(prev_cached + tokens_cached))

    async def get_daily_tokens(self) -> dict:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stored_date = await self.get("daily_tokens_date", "")
        if stored_date != today:
            return {"date": today, "tokens_in": 0, "tokens_out": 0, "tokens_cached": 0}
        return {
            "date": today,
            "tokens_in":     int(await self.get("daily_tokens_in", "0") or "0"),
            "tokens_out":    int(await self.get("daily_tokens_out", "0") or "0"),
            "tokens_cached": int(await self.get("daily_tokens_cached", "0") or "0"),
        }

    async def ensure_strategy_populated(self, default_strategy: str) -> None:
        """If strategy is empty or very short, populate with default_strategy."""
        current = await self.get_strategy()
        if len(current.strip()) < 200:
            await self.set_strategy(default_strategy)
            log.info("DB: strategy initialized with default template (%d chars)", len(default_strategy))

    async def recent_trades(self, limit: int = 10) -> list[Trade]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """
                SELECT id, ts, market_id, market_title, token_id, side,
                       size_usdc, price, status, dry_run, reasoning
                FROM trades ORDER BY ts DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        return [
            Trade(
                id=r[0], ts=r[1], market_id=r[2], market_title=r[3],
                token_id=r[4], side=r[5], size_usdc=r[6], price=r[7],
                status=r[8], dry_run=bool(r[9]), reasoning=r[10],
            )
            for r in rows
        ]
