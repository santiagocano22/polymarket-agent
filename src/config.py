"""Configuration loaded from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from eth_account import Account

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value == "a":
        raise RuntimeError(
            f"Missing required env var: {name}. Fill it in .env before starting."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Config:
    polymarket_private_key: str
    polymarket_api_key: str
    polymarket_api_secret: str
    polymarket_api_passphrase: str
    polymarket_wallet_address: str
    polymarket_chain_id: int
    polymarket_signature_type: int
    polymarket_funder: str
    eoa_address: str

    anthropic_api_key: str
    claude_model: str

    telegram_bot_token: str
    telegram_chat_id: int

    db_path: str
    loop_interval_seconds: int
    max_markets_per_cycle: int
    market_min_volume_24h: float
    market_max_volume_24h: float
    market_max_days_to_resolution: int

    trading_hour_start_utc: int   # hora UTC en que el bot empieza a operar
    trading_hour_end_utc: int     # hora UTC en que el bot deja de operar

    dry_run: bool

    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_api_host: str = "https://data-api.polymarket.com"


def load_config() -> Config:
    pk_raw = _require("POLYMARKET_PRIVATE_KEY")
    pk = pk_raw if pk_raw.startswith("0x") else f"0x{pk_raw}"
    eoa = Account.from_key(pk).address

    wallet = _require("POLYMARKET_WALLET_ADDRESS")

    sig_type_env = _optional("POLYMARKET_SIGNATURE_TYPE")
    if sig_type_env:
        sig_type = int(sig_type_env)
    elif eoa.lower() == wallet.lower():
        sig_type = 0
    else:
        sig_type = 2  # POLY_GNOSIS_SAFE (default for web-onboarded accounts)

    funder = wallet if sig_type != 0 else ""

    return Config(
        polymarket_private_key=pk,
        polymarket_api_key=_require("POLYMARKET_API_KEY"),
        polymarket_api_secret=_require("POLYMARKET_API_SECRET"),
        polymarket_api_passphrase=_require("POLYMARKET_API_PASSPHRASE"),
        polymarket_wallet_address=wallet,
        polymarket_chain_id=int(_optional("POLYMARKET_CHAIN_ID", "137")),
        polymarket_signature_type=sig_type,
        polymarket_funder=funder,
        eoa_address=eoa,
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        claude_model=_optional("CLAUDE_MODEL", "claude-sonnet-4-6"),
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=int(_require("TELEGRAM_CHAT_ID")),
        db_path=_optional("DB_PATH", "data/agent.db"),
        loop_interval_seconds=int(_optional("LOOP_INTERVAL_SECONDS", "300")),
        max_markets_per_cycle=int(_optional("MAX_MARKETS_PER_CYCLE", "20")),
        market_min_volume_24h=float(_optional("MARKET_MIN_VOLUME_24H", "500")),
        market_max_volume_24h=float(_optional("MARKET_MAX_VOLUME_24H", "10000000")),
        market_max_days_to_resolution=int(_optional("MARKET_MAX_DAYS_TO_RESOLUTION", "2")),
        trading_hour_start_utc=int(_optional("TRADING_HOUR_START_UTC", "14")),
        trading_hour_end_utc=int(_optional("TRADING_HOUR_END_UTC", "22")),
        dry_run=_optional("DRY_RUN", "true").lower() in ("1", "true", "yes"),
    )
