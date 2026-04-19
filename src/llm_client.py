"""LLM decision engine (Anthropic / Claude)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from .config import Config
from .polymarket_client import Market, Position

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an autonomous prediction-market trading agent on Polymarket.

EXECUTION MODEL:
- All orders are LIMIT orders (maker, 0% fee on Geopolitical markets). Never market orders.
- For BUY: set limit_price at or 1 cent below the current outcome_price to guarantee maker status.
  Example: if YES price is 0.72, set limit_price=0.71 or 0.72.
- For SELL: set limit_price at or 1 cent above the current price of your held token.
- size_usdc is the USDC you want to spend (BUY) or liquidate (SELL). The system converts to shares.
- shares = size_usdc / limit_price.

DECISION FRAMEWORK (apply before every action):
1. Estimate p_real from base rates, recent news, Metaculus/Manifold cross-check.
2. Compute edge = p_real − market_price. Minimum edge to act: 3%.
3. Size with Quarter-Kelly capped at 8% of bankroll: f = 0.25 × edge / (1 − market_price).
4. Verify the market has adequate liquidity (spread < 10¢, sufficient depth).
5. Define invalidation condition: what event would make you exit.

OUTPUT per decision: include p_real estimate, 3 sources used, edge, Kelly size, invalidation condition in the reasoning field.

HARD LIMITS (enforced by code — do not exceed):
- Minimum order: 5 shares. Minimum size_usdc = 5 × limit_price (e.g. $0.50 at price 0.10, $4.50 at price 0.90).
- max_per_trade_usd and max_exposure_usd are absolute ceilings.
- Zero decisions is always valid. Only act when you have genuine edge.
"""


DECISION_TOOL = {
    "name": "submit_decisions",
    "description": "Submit trading decisions for this cycle. All orders are limit (maker).",
    "input_schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Market overview and phase status (1-3 sentences).",
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "market_id":    {"type": "string"},
                        "market_title": {"type": "string"},
                        "token_id": {
                            "type": "string",
                            "description": "CLOB token id of YES or NO outcome to trade.",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["BUY", "SELL", "SKIP"],
                        },
                        "limit_price": {
                            "type": "number",
                            "description": (
                                "Limit price (0-1). For BUY: at or 1¢ below current price. "
                                "For SELL: at or 1¢ above current price. Required for BUY/SELL."
                            ),
                        },
                        "size_usdc": {
                            "type": "number",
                            "description": "USDC to spend (BUY) or liquidate (SELL). 0 for SKIP.",
                        },
                        "p_real": {
                            "type": "number",
                            "description": "Your estimated true probability (0-1).",
                        },
                        "edge": {
                            "type": "number",
                            "description": "p_real − market_price.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Sources used, edge calculation, Kelly sizing, "
                                "invalidation condition."
                            ),
                        },
                    },
                    "required": ["market_id", "token_id", "action", "reasoning"],
                },
            },
        },
        "required": ["analysis", "decisions"],
    },
}


@dataclass
class Decision:
    market_id: str
    market_title: str
    token_id: str
    action: str       # "BUY" | "SELL" | "SKIP"
    limit_price: float
    size_usdc: float
    p_real: float
    edge: float
    reasoning: str


@dataclass
class LLMResult:
    analysis: str
    decisions: list[Decision]


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = AsyncAnthropic(api_key=cfg.anthropic_api_key)

    async def decide(
        self,
        *,
        strategy: str,
        markets: list[Market],
        positions: list[Position],
        usdc_balance: float,
        max_exposure_usd: float,
        max_per_trade_usd: float | None,
    ) -> LLMResult:
        user_msg = _build_user_message(
            strategy=strategy,
            markets=markets,
            positions=positions,
            usdc_balance=usdc_balance,
            max_exposure_usd=max_exposure_usd,
            max_per_trade_usd=max_per_trade_usd,
        )

        resp = await self._client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=4096,
            # System prompt + tool schema are static — cache them for up to 1 hour.
            # Saves ~90% on those tokens once the cache is warm.
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": "submit_decisions"},
            messages=[{"role": "user", "content": user_msg}],
        )
        # Log cache hit/miss for cost visibility
        u = resp.usage
        cached_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cached_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        log.debug(
            "LLM usage — input: %d  cache_read: %d  cache_write: %d  output: %d",
            u.input_tokens, cached_read, cached_write, u.output_tokens,
        )

        tool_use = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use is None:
            log.warning("LLM returned no tool_use: %s", resp.content)
            return LLMResult(analysis="(no tool_use returned)", decisions=[])

        payload = tool_use.input or {}
        decisions: list[Decision] = []
        for d in payload.get("decisions", []):
            try:
                decisions.append(
                    Decision(
                        market_id=str(d.get("market_id", "")),
                        market_title=str(d.get("market_title", "")),
                        token_id=str(d.get("token_id", "")),
                        action=str(d.get("action", "SKIP")).upper(),
                        limit_price=float(d.get("limit_price") or 0),
                        size_usdc=float(d.get("size_usdc") or 0),
                        p_real=float(d.get("p_real") or 0),
                        edge=float(d.get("edge") or 0),
                        reasoning=str(d.get("reasoning", "")),
                    )
                )
            except Exception as e:
                log.warning("skipping malformed decision %s: %s", d, e)
        return LLMResult(
            analysis=str(payload.get("analysis", "")),
            decisions=decisions,
        )


def _build_user_message(
    *,
    strategy: str,
    markets: list[Market],
    positions: list[Position],
    usdc_balance: float,
    max_exposure_usd: float,
    max_per_trade_usd: float | None,
) -> str:
    market_blob = json.dumps(
        [m.to_llm_dict() for m in markets], ensure_ascii=False, indent=2
    )
    positions_blob = json.dumps(
        [
            {
                "market_id": p.market_id,
                "title": p.title,
                "outcome": p.outcome,
                "token_id": p.token_id,
                "shares": round(p.size, 4),
                "avg_price": round(p.avg_price, 4),
                "current_value_usdc": round(p.current_value_usdc, 4),
            }
            for p in positions
        ],
        ensure_ascii=False,
        indent=2,
    )

    per_trade = f"${max_per_trade_usd:.2f}" if max_per_trade_usd else "dynamic (10% bankroll)"

    return f"""USER STRATEGY:
---
{strategy.strip() or "(no strategy set — return zero decisions)"}
---

ACCOUNT STATE:
- USDC balance: ${usdc_balance:.2f}
- Hard limits: max_exposure=${max_exposure_usd:.2f}, max_per_trade={per_trade}

CURRENT POSITIONS:
{positions_blob}

CANDIDATE MARKETS (pre-filtered: vol $50k-$500k, resolution <30 days):
{market_blob}

Call submit_decisions. Zero decisions is valid if no genuine edge exists today."""
