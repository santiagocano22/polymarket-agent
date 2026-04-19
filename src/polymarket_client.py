"""Wrapper around py-clob-client + Polymarket Gamma/Data APIs."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Market:
    """Compact market representation used by the LLM."""
    id: str
    slug: str
    question: str
    description: str
    end_date: str | None
    volume_24h: float
    liquidity: float
    outcomes: list[str]              # e.g. ["Yes", "No"]
    outcome_prices: list[float]      # aligned with outcomes
    clob_token_ids: list[str]        # YES/NO token ids aligned with outcomes

    def yes_token_id(self) -> str:
        return self.clob_token_ids[0]

    def no_token_id(self) -> str:
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else ""

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "description": (self.description or "")[:400],
            "end_date": self.end_date,
            "volume_24h": round(self.volume_24h, 2),
            "liquidity": round(self.liquidity, 2),
            "outcomes": self.outcomes,
            "outcome_prices": [round(p, 4) for p in self.outcome_prices],
            "yes_token_id": self.yes_token_id(),
            "no_token_id": self.no_token_id(),
        }


@dataclass
class Position:
    market_id: str            # conditionId
    token_id: str
    outcome: str              # "Yes" / "No"
    size: float               # shares held
    avg_price: float
    current_value_usdc: float
    title: str


class PolymarketClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._clob: ClobClient | None = None
        self._http = httpx.AsyncClient(timeout=20.0)

    def _clob_client(self) -> ClobClient:
        if self._clob is not None:
            return self._clob
        creds = ApiCreds(
            api_key=self.cfg.polymarket_api_key,
            api_secret=self.cfg.polymarket_api_secret,
            api_passphrase=self.cfg.polymarket_api_passphrase,
        )
        kwargs: dict[str, Any] = dict(
            host=self.cfg.clob_host,
            key=self.cfg.polymarket_private_key,
            chain_id=self.cfg.polymarket_chain_id,
            creds=creds,
            signature_type=self.cfg.polymarket_signature_type,
        )
        if self.cfg.polymarket_funder:
            kwargs["funder"] = self.cfg.polymarket_funder
        self._clob = ClobClient(**kwargs)
        return self._clob

    async def close(self) -> None:
        await self._http.aclose()

    # ---------------------------------------------------------------- markets
    async def search_markets(
        self,
        *,
        limit: int = 60,
        min_volume_24h: float = 50_000,
        max_volume_24h: float = 500_000,
        max_days_to_resolution: int = 30,
    ) -> list[Market]:
        """Active markets filtered by volume range and resolution window."""
        params = {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": str(limit),
            "order": "volume24hr",
            "ascending": "false",
        }
        r = await self._http.get(
            f"{self.cfg.gamma_host}/markets", params=params
        )
        r.raise_for_status()
        data = r.json()
        markets = [m for raw in data if (m := _parse_market(raw)) is not None]

        cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=max_days_to_resolution)
        filtered: list[Market] = []
        for m in markets:
            if not (min_volume_24h <= m.volume_24h <= max_volume_24h):
                continue
            if m.end_date:
                try:
                    end = dt.datetime.fromisoformat(
                        m.end_date.replace("Z", "+00:00")
                    )
                    if end > cutoff:
                        continue
                except Exception:
                    pass
            filtered.append(m)
        return filtered

    # ---------------------------------------------------------------- balance
    async def get_usdc_balance(self) -> float:
        def _call() -> float:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = self._clob_client().get_balance_allowance(params)
            # Balance in USDC has 6 decimals.
            return float(resp.get("balance", "0")) / 1_000_000
        return await asyncio.to_thread(_call)

    # -------------------------------------------------------------- positions
    async def get_positions(self) -> list[Position]:
        params = {
            "user": self.cfg.polymarket_wallet_address,
            "sizeThreshold": "0.01",
        }
        try:
            r = await self._http.get(
                f"{self.cfg.data_api_host}/positions", params=params
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("positions fetch failed: %s", e)
            return []
        out: list[Position] = []
        for p in data:
            try:
                out.append(
                    Position(
                        market_id=str(p.get("conditionId", "")),
                        token_id=str(p.get("asset", "")),
                        outcome=str(p.get("outcome", "")),
                        size=float(p.get("size", 0)),
                        avg_price=float(p.get("avgPrice", 0)),
                        current_value_usdc=float(p.get("currentValue", 0)),
                        title=str(p.get("title", "")),
                    )
                )
            except Exception as e:
                log.debug("skipping malformed position %s: %s", p, e)
        return out

    # ----------------------------------------------------------------- orders
    MIN_SHARES = 5.0  # Polymarket CLOB minimum order size

    async def buy_limit(
        self, token_id: str, limit_price: float, usdc_amount: float
    ) -> dict[str, Any]:
        """Place a maker BUY limit order.
        Enforces minimum 5 shares; returns error dict if budget is insufficient."""
        def _call() -> dict[str, Any]:
            from decimal import Decimal, ROUND_DOWN
            client = self._clob_client()
            price_d = Decimal(str(max(0.001, min(0.999, round(float(limit_price), 4)))))
            usdc_d  = Decimal(str(usdc_amount))

            # Polymarket CLOB uses full-collateral: requires 1 USDC per share
            # regardless of price.  So shares must NEVER exceed usdc_amount.
            # We want as many shares as the price allows, but capped at usdc_amount:
            #   desired  = usdc / price  (> usdc when price < 1)
            #   allowed  = min(desired, usdc)  → always ≤ usdc
            desired  = usdc_d / price_d
            allowed  = min(desired, usdc_d)
            shares_d = allowed.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            shares   = float(max(Decimal(str(self.MIN_SHARES)), shares_d))

            args = OrderArgs(token_id=token_id, price=float(price_d), size=shares, side="BUY")
            signed = client.create_order(args)
            return client.post_order(signed, OrderType.GTC)
        return await asyncio.to_thread(_call)

    async def sell_limit(
        self, token_id: str, limit_price: float, shares: float
    ) -> dict[str, Any]:
        """Place a maker SELL limit order. Enforces min 5 shares and valid price range."""
        def _call() -> dict[str, Any]:
            client = self._clob_client()
            price = max(0.001, min(0.999, round(float(limit_price), 4)))
            actual_shares = max(self.MIN_SHARES, round(float(shares), 2))
            args = OrderArgs(token_id=token_id, price=price, size=actual_shares, side="SELL")
            signed = client.create_order(args)
            return client.post_order(signed, OrderType.GTC)
        return await asyncio.to_thread(_call)

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open limit orders to free up locked USDC."""
        def _call() -> dict[str, Any]:
            try:
                return self._clob_client().cancel_all()
            except Exception as e:
                return {"error": str(e)}
        return await asyncio.to_thread(_call)

    async def get_open_orders(self) -> list[dict]:
        """Return list of currently open orders."""
        def _call() -> list[dict]:
            try:
                resp = self._clob_client().get_orders()
                return resp if isinstance(resp, list) else []
            except Exception as e:
                log.warning("get_open_orders error: %s", e)
                return []
        return await asyncio.to_thread(_call)

    async def get_midprice(self, token_id: str) -> float | None:
        def _call() -> float | None:
            try:
                resp = self._clob_client().get_midpoint(token_id)
                mid = resp.get("mid") if isinstance(resp, dict) else None
                return float(mid) if mid is not None else None
            except Exception as e:
                log.debug("midpoint error for %s: %s", token_id, e)
                return None
        return await asyncio.to_thread(_call)


# ----------------------------------------------------------------- helpers
def _parse_market(raw: dict[str, Any]) -> Market | None:
    try:
        outcomes = _maybe_json_list(raw.get("outcomes"))
        prices = _maybe_json_list(raw.get("outcomePrices"))
        token_ids = _maybe_json_list(raw.get("clobTokenIds"))
        if not outcomes or not token_ids:
            return None
        return Market(
            id=str(raw.get("conditionId") or raw.get("id") or ""),
            slug=str(raw.get("slug") or ""),
            question=str(raw.get("question") or ""),
            description=str(raw.get("description") or ""),
            end_date=raw.get("endDate"),
            volume_24h=float(raw.get("volume24hr") or 0),
            liquidity=float(raw.get("liquidity") or 0),
            outcomes=[str(o) for o in outcomes],
            outcome_prices=[float(p) for p in prices] if prices else [],
            clob_token_ids=[str(t) for t in token_ids],
        )
    except Exception as e:
        log.debug("skipping malformed market: %s", e)
        return None


def _maybe_json_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []
