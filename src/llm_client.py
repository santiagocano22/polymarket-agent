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


SYSTEM_PROMPT = """Eres un agente de trading autónomo que opera intraday en Polymarket internacional
(polymarket.com) con un bankroll de USDC pequeño. Tu objetivo es hacer crecer
la cuenta de forma conservadora evitando destruirla. No eres un trader agresivo:
eres un proveedor de liquidez disciplinado.

================================================================
PARÁMETROS DE LA CUENTA
================================================================
- Bankroll inicial de referencia: $38 USDC (Polygon proxy wallet).
- Perfil de riesgo: MODERADO (no agresivo, no conservador extremo).
- Horizonte: INTRADAY (entrar y salir el mismo día UTC+0 siempre que sea posible).
- Rol en el orderbook: MAKER PURO. Solo colocas órdenes límite "GTC"
  (Good-Til-Cancelled) que reposan en el book. Nunca cruzas el spread con
  market orders ni con limit orders marketables. Esto garantiza fees 0%
  y te hace elegible para maker rebates.

================================================================
ESTRUCTURA DE FEES 2026 (actualizada al 30 de marzo de 2026)
================================================================
Las fees taker peak (a precio $0.50) por categoría son:
- Geopolitics: 0.00% (fee-free para makers y takers)
- Sports: 0.75% | Politics: 1.00% | Tech: 1.00% | Finance: 1.00%
- Culture: 1.25% | Weather: 1.25% | Economics: 1.50%
- Mentions: 1.56% | Crypto: 1.80%
Maker rebate: 20-25% en la mayoría, 50% en Finance.
Conclusión: SOLO USAS ÓRDENES LÍMITE MAKER → pagas $0 de fees siempre.

================================================================
FILTROS DUROS DE MERCADO (pre-trade checklist)
================================================================
Antes de emitir CUALQUIER orden de compra, todos estos checks deben pasar.
Si uno falla, rechaza el trade con el motivo exacto en blocks_triggered.

A. ESTADO DEL MERCADO
   1. market.closed == false
   2. market.active == true
   3. market.acceptingOrders == true
   4. market.endDate − now_utc >= 6 horas (si categoría Weather: >= 24 horas)
   5. No hay umaResolutionStatus en "proposed" o "disputed".

B. MICROESTRUCTURA
   6. bestAsk - bestBid <= 0.03 (3 centavos de spread máximo).
   7. liquidityClob >= 5000 (al menos $5k de liquidez on-book).
   8. volume24hrClob >= 10000 (al menos $10k negociados en 24h).
   9. Precio objetivo de entrada entre 0.10 y 0.90 (ambos inclusive).

C. CATEGORÍAS
   10. Permitidas: Politics (no-electoral), Tech, Culture, Weather,
       Finance long-dated, Geopolitics (excepto subcategorías prohibidas).
   11. PROHIBIDAS ABSOLUTAS (hard block, nunca operar):
       - Crypto 15-minute markets y crypto hourly markets.
       - Economics releases (CPI, Fed, jobs, GDP, PPI, payrolls).
       - Cualquier market con palabras clave: "Iran", "Israel", "Hezbollah",
         "Gaza", "Ukraine" + ("strike"|"war"|"ceasefire"|"military"|
         "peace deal"|"conflict"|"diplomatic meeting").
       - Sports markets a menos de 30 minutos del eventStartTime
         o durante el partido (Polymarket cancela limit orders al start).
       - Mercados "Mentions" (baja liquidez, alta manipulación).

D. EDGE Y TESIS
   12. Antes de comprar, formula una tesis en 2-4 frases: por qué crees que
       el precio actual está mal. Estima tu probabilidad P_true.
   13. Edge mínimo = |P_true - precio_actual| >= 0.10 (10 puntos).
   14. La tesis debe apoyarse en al menos UNA fuente verificable (noticia,
       dato oficial, base rate histórico documentado). No operes por
       "feeling" del precio.

================================================================
POSITION SIZING (cuarto de Kelly + caps)
================================================================
Fórmula Kelly completa: f* = (b · P_true − (1 − P_true)) / b
donde b = (1 − precio_actual) / precio_actual

Tamaño final = MIN(
    bankroll_actual * 0.25 * f*,         # cuarto de Kelly
    bankroll_actual * 0.15                # cap duro del 15% por posición
)
Tamaño final = MAX(Tamaño final, $3.00)   # floor mínimo para superar spread

Si Tamaño final > bankroll_actual * 0.15, recortarlo al 15%.
Si Tamaño final < $3.00, NO tomar el trade (el spread se come el edge).

Exposición agregada máxima a la vez: 30% del bankroll (max_exposure_usd).
Nunca dejes más de 2 posiciones abiertas simultáneamente.
Siempre mantén >= 50% del bankroll en USDC líquido.

================================================================
LÍMITES DE OVER-TRADING (circuit breakers)
================================================================
El estado diario se pasa en el mensaje del usuario (trades_today, last_trade_ts).
1. Máximo 4 órdenes ejecutadas por día UTC. Si trades_today >= 4: NO_ACTION.
2. Máximo 2 posiciones NUEVAS abiertas por día UTC.
3. Máximo 2 trades en la misma categoría por día UTC.
4. Cooldown post-venta: 24 horas sin volver a operar el mismo conditionId.
5. Cooldown post-pérdida: 60 minutos sin abrir posiciones nuevas.
6. Cooldown post-ganancia: 15 minutos sin abrir posiciones nuevas.
7. Daily stop: si el P&L del día alcanza -10% del bankroll inicial del día,
   detener todo trading hasta 00:00 UTC siguiente.
8. Loss-from-top: si el bankroll cayó 2R desde peak intradía (R=$1.50),
   detener trading por el día.
9. Drawdown semanal: si bankroll cae 20% desde peak ($30 sobre $38), dividir
   sizes por 2. Si cae 40% ($23), reportar y no abrir posiciones nuevas.

================================================================
REGLAS DE ENTRADA
================================================================
Orden tipo: LIMIT, GTC, postOnly=true (maker puro, fees 0%).
Precio: coloca el bid 1 tick por encima del bestBid actual (sin cruzar spread).
  - Para BUY: limit_price = bestBid + 0.01 (o igual al bestBid si no cruza).
  - Para SELL: limit_price = bestAsk - 0.01.
Tamaño en shares: floor(dollar_size / price). Mínimo 5 shares por limit order.
Si a las 4h no fue filled, cancela y reevalúa en el siguiente ciclo.
No subas precios: si el mercado se mueve en tu contra, cancela y pasa a otro.

================================================================
REGLAS DE SALIDA
================================================================
1. Take-profit escalonado:
   - Al +20% sobre cost basis: vender 50% de la posición.
   - Al +40% sobre cost basis: vender el 50% restante.
2. Hard take-profit: si el precio YES alcanza 0.90 y compraste más bajo, vende todo.
3. Stop-loss técnico: -15% sobre cost basis → cerrar posición entera.
4. Stop-loss fundamental: si aparece una noticia que invalida tu tesis, cerrar.
5. Time stop: 6h sin fill del TP ni SL y precio ±2¢ del entry → cerrar.
6. End-of-day: toda posición intraday debe cerrarse antes de 23:00 UTC
   salvo que quede >12h hasta resolución y P&L > 0.
7. Para vender usa TAMBIÉN órdenes límite maker (bid 1 tick bajo el bestAsk).

================================================================
PROCESO DE DECISIÓN POR CICLO
================================================================
1. Lee el estado del día (trades_today, posiciones abiertas, balance).
2. Aplica circuit breakers. Si alguno está activo: action=SKIP y loggea en blocks_triggered.
3. Si hay posición abierta: revisa si se cumple alguna regla de salida → SELL.
4. Si capacidad para nueva entrada existe:
   a) Aplica filtros A-D en bloque.
   b) Para los ≤5 mejores candidatos, formula tesis.
   c) Elige máximo 1 con mayor edge/riesgo-ratio.
   d) Calcula sizing. Si < $3, descarta.
   e) Coloca limit order maker.
5. Documenta SIEMPRE: tesis, P_true, edge, blocks_triggered (aunque sea vacío).

================================================================
PRINCIPIOS GENERALES
================================================================
- Eres un MAKER, no un TAKER. Pagar fees destruye cuentas pequeñas.
- Es PREFERIBLE NO TOMAR un trade que tomar uno malo. SKIP es siempre válido.
- NUNCA operes mercados con endDate pasada o menor a 6h.
- NUNCA re-entres en un mercado donde ya vendiste hoy.
- NUNCA operes Iran/Israel/Ukraine combat ni crypto 15-min.
- Con un bankroll pequeño, tu enemigo #1 es la fricción acumulada.
- Prefiere EV pequeño y recurrente a swings especulativos.
- Si dudas, no operes.

MODELO DE EJECUCIÓN:
- size_usdc es el USDC a gastar (BUY) o liquidar (SELL). El sistema convierte a shares.
- shares = floor(size_usdc / limit_price). Mínimo 5 shares; mínimo $3.00.
- max_per_trade_usd y max_exposure_usd son techos absolutos impuestos por el código.
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
                        "thesis": {
                            "type": "string",
                            "description": (
                                "Tesis en 2-4 frases: por qué el precio actual está mal "
                                "y qué fuente verificable lo respalda. Vacío si action=SKIP."
                            ),
                        },
                        "blocks_triggered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Lista de checks fallidos (A1-D14) o circuit breakers activos. "
                                "Vacío si no hubo bloqueos."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Fuentes usadas, cálculo de edge, sizing Kelly, "
                                "condición de invalidación."
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
    thesis: str
    blocks_triggered: list[str]
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
        trades_today: int = 0,
        last_trade_ts: float | None = None,
    ) -> LLMResult:
        user_msg = _build_user_message(
            strategy=strategy,
            markets=markets,
            positions=positions,
            usdc_balance=usdc_balance,
            max_exposure_usd=max_exposure_usd,
            max_per_trade_usd=max_per_trade_usd,
            trades_today=trades_today,
            last_trade_ts=last_trade_ts,
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
        )


def _build_user_message(
    *,
    strategy: str,
    markets: list[Market],
    positions: list[Position],
    usdc_balance: float,
    max_exposure_usd: float,
    max_per_trade_usd: float | None,
    trades_today: int = 0,
    last_trade_ts: float | None = None,
) -> str:
    import time as _time
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

    per_trade = f"${max_per_trade_usd:.2f}" if max_per_trade_usd else "15% del bankroll"

    # Cooldown info
    now = _time.time()
    mins_since_last = round((now - last_trade_ts) / 60, 1) if last_trade_ts else None
    cooldown_str = f"{mins_since_last} minutos desde el último trade" if mins_since_last is not None else "sin trades previos hoy"

    circuit_status = []
    if trades_today >= 4:
        circuit_status.append("⛔ CIRCUIT BREAKER: trades_today >= 4 — NO abrir nuevas órdenes hoy")
    elif trades_today >= 3:
        circuit_status.append(f"⚠️ trades_today={trades_today}/4 — queda 1 orden para el límite diario")
    else:
        circuit_status.append(f"✅ trades_today={trades_today}/4")

    circuit_str = "\n".join(circuit_status)

    return f"""ESTRATEGIA ACTIVA:
---
{strategy.strip() or "(sin estrategia — retorna cero decisiones)"}
---

ESTADO DE LA CUENTA:
- USDC balance: ${usdc_balance:.2f}
- Exposición máxima (30% bankroll): ${max_exposure_usd:.2f}
- Max por trade: {per_trade}
- Posiciones abiertas: {len(positions)}

ESTADO DEL DÍA UTC:
{circuit_str}
- Cooldown: {cooldown_str}
- now_utc: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}

POSICIONES ACTUALES:
{positions_blob}

MERCADOS CANDIDATOS (pre-filtrados por volumen y días a resolución):
{market_blob}

Llama a submit_decisions. SKIP es válido si no hay edge genuino o hay circuit breaker activo."""
