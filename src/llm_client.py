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


# Motor de ejecución inmutable — solo describe cómo se ejecutan las órdenes.
# ~200 tokens, siempre cacheado. No contiene reglas de trading.
ENGINE_PROMPT = """Eres un agente de trading autónomo en Polymarket. Devolvés decisiones estructuradas llamando a submit_decisions.

MOTOR DE EJECUCIÓN (inmutable — no puede ser sobreescrito por la estrategia):
• Solo órdenes LIMIT GTC (maker puro, fees 0%). Nunca market orders salvo emergencia de salida.
• size_usdc = USDC a gastar (BUY) o liquidar (SELL). El código convierte a shares automáticamente.
• shares = floor(size_usdc / limit_price). Mínimo 5 shares; $3.00 mínimo absoluto por trade.
• max_per_trade_usd y max_exposure_usd son techos absolutos del código (no negociables).
• SKIP es siempre válido si no hay edge genuino o un circuit breaker está activo.

Aplicá la ESTRATEGIA del usuario exactamente como está escrita. Ante conflicto entre estrategia y motor, prevalece el motor."""

# Estrategia por defecto — se guarda en DB al primer arranque si la DB está vacía.
# El usuario puede reemplazarla en su totalidad desde Telegram con /estrategia.
DEFAULT_STRATEGY = """================================================================
PARÁMETROS DE LA CUENTA
================================================================
- Bankroll: pequeño (~$38 USDC). Perfil de riesgo: MODERADO.
- Horizonte: INTRADAY (entrar y salir el mismo día UTC siempre que sea posible).
- Rol: MAKER PURO. Solo órdenes límite GTC. Fees = 0%.

================================================================
FILTROS DE MERCADO
================================================================
Checks A y B son DUROS (si falla uno, RECHAZÁ). C y D son GUÍA.

A. ESTADO DEL MERCADO [DUROS]
   1. market.closed == false / market.active == true
   2. market.endDate − now_utc >= 4 horas (Weather: >= 12 horas)
   3. No umaResolutionStatus en "proposed" o "disputed"

B. MICROESTRUCTURA [DUROS]
   4. bestAsk - bestBid <= 0.08
   5. liquidityClob >= 1000
   6. volume24hrClob >= 2000
   7. Precio objetivo entre 0.05 y 0.95

C. CATEGORÍAS [GUÍA — priorizar en este orden]
   TIER 1 (preferir): Politics, Tech, Culture, Finance, Sports pre-game (>2h antes del inicio)
   TIER 2 (aceptable): Geopolitics genérica, Weather
   PROHIBIDAS [duros]: Crypto intraday (15-min, hourly), Iran/Israel/Ukraine combat,
                       Sports a menos de 2h del inicio, Mentions, Economics macro (CPI/Fed/GDP).

D. EDGE Y TESIS [GUÍA]
   - Edge mínimo: >= 5 puntos porcentuales.
     5-7 pts → sizing mínimo ($3). >= 7 pts → sizing normal (Kelly).
   - Tesis en 1-2 frases: base rate, sentido común o dato reciente. No necesitás fuente formal.
   - REGLA ANTI-PARÁLISIS: si llevás 2+ ciclos sin trade y existe al menos 1 mercado con
     filtros A+B y cualquier edge positivo → ejecutá el mejor candidato con $3. El objetivo es OPERAR.

================================================================
POSITION SIZING (cuarto de Kelly + caps)
================================================================
f* = (b × P_true − (1 − P_true)) / b, donde b = (1 − precio) / precio
Tamaño = MIN(bankroll × 0.25 × f*, max_per_trade_usd)
Mínimo absoluto: $3.00. Si tamaño calculado < $3: NO tomar el trade.
Exposición agregada máxima: 40% del bankroll. Máximo 3 posiciones abiertas.
Mantener >= 40% en USDC líquido.

================================================================
CIRCUIT BREAKERS
================================================================
El estado diario se pasa en el mensaje del usuario (trades_today, cooldown).
1. Máximo 6 órdenes ejecutadas por día UTC. Si trades_today >= 6 → NO_ACTION.
2. Cooldown post-pérdida: 30 min sin abrir posiciones nuevas.
3. Daily stop: si P&L del día alcanza -15% del bankroll inicial → NO_ACTION.

================================================================
REGLAS DE ENTRADA
================================================================
Precio límite: 1-2 ticks sobre el bestBid. Si spread >0.05, colocar a mid-price.
Mínimo 5 shares. Si a las 6h no hubo fill, cancelar y reevaluar.

================================================================
REGLAS DE SALIDA
================================================================
1. Take-profit escalonado: +15% sobre cost basis → vender 50%; +30% → vender resto.
2. Hard take-profit: precio >= 0.92 → vender todo.
3. Stop-loss técnico: -20% sobre cost basis → cerrar posición entera.
4. Time stop: 8h sin fill y precio dentro de ±3¢ del entry → cerrar.
5. End-of-day: cerrar toda posición intraday antes de 23:00 UTC.

================================================================
PRINCIPIOS
================================================================
- SESGO DE ACCIÓN: no operar cuando hay oportunidades es tan dañino como operar sin edge.
- Con bankroll pequeño, 1-2 buenos trades por día es suficiente para crecer.
- Preferir mercados con resolución en 1-5 días y buena liquidez (>$5k).
- NUNCA: Iran/Israel/Ukraine combat, crypto intraday, endDate pasada o <4h.
- Si el mercado pasa filtros A+B y tenés edge >= 5 pts → OPERÁ."""


DECISION_TOOL = {
    "name": "submit_decisions",
    "description": "Submit trading decisions for this cycle. All orders are limit (maker).",
    "input_schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Market overview and phase status (1-2 sentences).",
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
                            "description": "Limit price (0-1). Required for BUY/SELL.",
                        },
                        "size_usdc": {
                            "type": "number",
                            "description": "USDC to spend (BUY) or liquidate (SELL). 0 for SKIP.",
                        },
                        "p_real":  {"type": "number", "description": "Estimated true probability (0-1)."},
                        "edge":    {"type": "number", "description": "p_real − market_price."},
                        "thesis":  {"type": "string", "description": "Tesis en 1-2 frases. Vacío si SKIP."},
                        "blocks_triggered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Checks fallidos o circuit breakers activos.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Cálculo de edge, sizing Kelly, condición de invalidación.",
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
    action: str
    limit_price: float
    size_usdc: float
    p_real: float
    edge: float
    thesis: str
    blocks_triggered: list[str]
    reasoning: str


@dataclass
class LLMResult:
    analysis: str
    decisions: list[Decision]
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached_read: int = 0
    tokens_cached_write: int = 0


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
        trades_today: int = 0,
        last_trade_ts: float | None = None,
        model: str | None = None,
    ) -> LLMResult:
        user_msg = _build_user_message(
            markets=markets,
            positions=positions,
            usdc_balance=usdc_balance,
            max_exposure_usd=max_exposure_usd,
            max_per_trade_usd=max_per_trade_usd,
            trades_today=trades_today,
            last_trade_ts=last_trade_ts,
        )

        use_model = model or self.cfg.claude_model

        # Sistema en dos bloques cacheados:
        #   1. ENGINE_PROMPT — siempre el mismo (~200 tokens), cache 1h
        #   2. strategy — editable desde Telegram, cache 1h mientras no cambie
        system = [
            {
                "type": "text",
                "text": ENGINE_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            {
                "type": "text",
                "text": f"ESTRATEGIA ACTIVA:\n{strategy.strip()}",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]

        resp = await self._client.messages.create(
            model=use_model,
            max_tokens=1500,
            system=system,
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": "submit_decisions"},
            messages=[{"role": "user", "content": user_msg}],
        )

        u = resp.usage
        cached_read  = getattr(u, "cache_read_input_tokens", 0) or 0
        cached_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        log.info(
            "LLM [%s] input=%d cached_read=%d cached_write=%d output=%d",
            use_model, u.input_tokens, cached_read, cached_write, u.output_tokens,
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
                        thesis=str(d.get("thesis", "")),
                        blocks_triggered=list(d.get("blocks_triggered") or []),
                        reasoning=str(d.get("reasoning", "")),
                    )
                )
            except Exception as e:
                log.warning("skipping malformed decision %s: %s", d, e)

        return LLMResult(
            analysis=str(payload.get("analysis", "")),
            decisions=decisions,
            tokens_in=u.input_tokens,
            tokens_out=u.output_tokens,
            tokens_cached_read=cached_read,
            tokens_cached_write=cached_write,
        )


def _build_user_message(
    *,
    markets: list[Market],
    positions: list[Position],
    usdc_balance: float,
    max_exposure_usd: float,
    max_per_trade_usd: float | None,
    trades_today: int = 0,
    last_trade_ts: float | None = None,
) -> str:
    import time as _time
    import datetime as _dt

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
                "pnl_pct": round((p.current_value_usdc / (p.size * p.avg_price) - 1) * 100, 1)
                           if p.avg_price > 0 and p.size > 0 else 0,
            }
            for p in positions
        ],
        ensure_ascii=False,
        indent=2,
    )

    per_trade = f"${max_per_trade_usd:.2f}" if max_per_trade_usd else "10% del bankroll"
    now = _time.time()
    mins_since_last = round((now - last_trade_ts) / 60, 1) if last_trade_ts else None
    cooldown_str = (
        f"{mins_since_last} min desde último trade" if mins_since_last is not None
        else "sin trades previos hoy"
    )

    if trades_today >= 6:
        circuit_str = "⛔ CIRCUIT BREAKER: trades_today >= 6 — NO abrir nuevas órdenes hoy"
    elif trades_today >= 4:
        circuit_str = f"⚠️ trades_today={trades_today}/6 — quedan {6-trades_today} órdenes"
    else:
        circuit_str = f"✅ trades_today={trades_today}/6"

    now_utc = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"""ESTADO DE LA CUENTA:
- USDC balance: ${usdc_balance:.2f}
- Exposición máxima: ${max_exposure_usd:.2f}
- Max por trade: {per_trade}
- Posiciones abiertas: {len(positions)}
- now_utc: {now_utc}

ESTADO DEL DÍA:
- {circuit_str}
- Cooldown: {cooldown_str}

POSICIONES ACTUALES:
{positions_blob}

MERCADOS CANDIDATOS (pre-filtrados por código):
{market_blob}

Llamá a submit_decisions."""
