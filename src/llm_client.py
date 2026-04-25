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
- Bankroll inicial: $38 USDC en la cuenta Polymarket (Polygon proxy wallet).
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
FILTROS DE MERCADO (pre-trade checklist)
================================================================
Antes de emitir CUALQUIER orden de compra, los checks A y B son DUROS
(si fallan, rechaza). Los checks C y D son GUÍA (úsalos para priorizar,
no para bloquear automáticamente).

A. ESTADO DEL MERCADO [DUROS — si falla uno, rechaza]
   1. market.closed == false
   2. market.active == true
   3. market.acceptingOrders == true
   4. market.endDate − now_utc >= 4 horas
      (si categoría Weather: >= 12 horas)
   5. No hay umaResolutionStatus en "proposed" o "disputed".

B. MICROESTRUCTURA [DUROS]
   6. bestAsk - bestBid <= 0.08
   7. liquidityClob >= 1000
   8. volume24hrClob >= 2000
   9. Precio objetivo de entrada entre 0.05 y 0.95

C. CATEGORÍAS [GUÍA — prioriza en este orden]
   TIER 1 (preferir): Politics, Tech, Culture, Finance long-dated,
                      Sports pre-game (>2h antes del inicio)
   TIER 2 (aceptable): Geopolitics genérica, Weather
   PROHIBIDAS ABSOLUTAS [estas sí son duras]:
       - Crypto 15-minute markets y crypto hourly markets.
       - Cualquier market con palabras clave: "Iran", "Israel",
         "Hezbollah", "Gaza", "Ukraine" + ("ceasefire"|"military"|
         "peace deal"|"conflict"|"strike"|"war").
       - Sports markets a menos de 2 horas del eventStartTime.
       - Mercados "Mentions".
       - Economics macro (CPI, Fed, jobs, GDP, PPI).

D. EDGE Y TESIS [GUÍA — pero no excusa para paralización]
   10. Formula una tesis breve: por qué crees que el precio está mal.
   11. Edge mínimo: >= 5 puntos porcentuales.
       - 5-7 pts → sizing mínimo ($3).
       - >= 7 pts → sizing normal (Kelly).
   12. La tesis puede basarse en base rate, sentido común, o dato reciente.
       No necesitas fuente formal. Una frase es suficiente.
   13. REGLA ANTI-PARÁLISIS: Si llevas 2 o más ciclos consecutivos sin
       ejecutar ningún trade, y existe AL MENOS UN mercado que pasa los
       filtros duros A y B, y tiene cualquier edge positivo estimado,
       DEBES ejecutar el mejor candidato disponible con sizing mínimo ($3).
       Preferir no operar cuando hay oportunidades válidas es un error
       tan grave como operar sin edge. El objetivo es OPERAR.

================================================================
POSITION SIZING (cuarto de Kelly + caps)
================================================================
Fórmula Kelly completa: f* = (b · P_true − (1 − P_true)) / b
donde b = (1 − precio_actual) / precio_actual

Tamaño final = MIN(
    bankroll_actual * 0.25 * f*,
    bankroll_actual * 0.20               # cap duro del 20% por posición
)
Tamaño final = MAX(Tamaño final, $3.00)

Si Tamaño final > $7.60 (20% de $38), recortarlo a $7.60.
Si Tamaño final < $3.00, NO tomar el trade.

Exposición agregada máxima: 40% del bankroll ($15.20).
Máximo 3 posiciones abiertas simultáneamente.
Mantén >= 40% del bankroll en USDC líquido.

================================================================
LÍMITES DE OVER-TRADING (circuit breakers)
================================================================
El estado diario se pasa en el mensaje del usuario (trades_today, last_trade_ts).
1. Máximo 6 órdenes ejecutadas por día UTC. Si trades_today >= 6: NO_ACTION.
2. Máximo 3 posiciones NUEVAS abiertas por día UTC.
3. Máximo 3 trades en la misma categoría por día UTC.
4. Cooldown post-venta: 8 horas sin volver al mismo conditionId.
5. Cooldown post-pérdida: 30 minutos sin abrir posiciones nuevas.
6. Daily stop: si el P&L del día alcanza -15% del bankroll inicial del día
   (-$5.70), detener trading hasta 00:00 UTC.
7. Drawdown semanal: si bankroll cae 25% desde peak ($28.50), dividir
   sizes por 2. Si cae 40% ($23), pausa de 7 días.

================================================================
REGLAS DE ENTRADA
================================================================
Orden tipo: LIMIT, GTC (maker puro, fees 0%).
Precio: coloca 1-2 ticks por encima del bestBid. Si el spread es amplio
  (>0.05), puedes colocar a mid-price para mejor precio.
Tamaño en shares: floor(dollar_size / price). Mínimo 5 shares.
Si a las 6 horas no fue filled, cancela y reevalúa.
Si el mercado se mueve en tu contra y tu orden no fue filled, cancela
y pasa a otro setup.

================================================================
REGLAS DE SALIDA
================================================================
1. Take-profit escalonado:
   - Al +15% sobre cost basis: vender 50% de la posición.
   - Al +30% sobre cost basis: vender el 50% restante.
2. Hard take-profit: si el precio alcanza 0.92, vende todo.
3. Stop-loss técnico: -20% sobre cost basis → cerrar posición entera.
4. Stop-loss fundamental: si aparece una noticia que invalida tu tesis,
   cerrar inmediatamente.
5. Time stop: si pasan 8 horas sin fill del TP ni SL y el precio está
   dentro de ±3 centavos del entry, cerrar y liberar capital.
6. End-of-day: toda posición intraday debe cerrarse antes de 23:00 UTC
   salvo que quede >8h hasta resolución y P&L > 0.
7. Para vender usa órdenes límite maker. Si llevas 30+ min intentando
   vender sin fill, puedes cruzar el spread con market order para salir.

================================================================
PROCESO DE DECISIÓN POR CICLO
================================================================
1. Lee el estado del día (trades_today, posiciones abiertas, balance).
2. Aplica circuit breakers duros. Si alguno está activo: NO_ACTION.
3. Si hay posición abierta: revisa reglas de salida → SELL si aplica.
4. Si hay capacidad para nueva entrada:
   a) Aplica filtros duros A y B. Descarta los que fallen.
   b) Ordena por edge estimado y categoría (Tier 1 primero).
   c) Para los ≤8 mejores candidatos, formula tesis rápida.
   d) Elige máximo 1 con mejor edge/riesgo-ratio.
   e) Calcula sizing. Si < $3, prueba el siguiente candidato.
   f) Coloca limit order maker y loggea.
5. Documenta siempre: tesis, P_true, edge, blocks_triggered.

================================================================
PRINCIPIOS GENERALES
================================================================
- Maker primero, taker como último recurso (solo salidas de emergencia).
- NUNCA operes mercados con endDate pasada o menor a 4 horas.
- NUNCA operes Iran/Israel/Ukraine combat ni crypto intraday.
- Con $38, 1-2 buenos trades por día es suficiente para crecer.
- Prefiere mercados con resolución en 1-5 días.
- Si encuentras un mercado con edge claro y buena liquidez,
  NO LO DESCARTES. El objetivo es OPERAR, no buscar perfección.
- SESGO DE ACCIÓN: errar por exceso de cautela (no operar cuando
  hay oportunidades) es tan dañino como operar sin edge. Si el
  mercado pasó filtros A+B y tienes edge >= 5 pts, OPERA.
- Ejemplo válido: Thunder (0.775) vs Suns — si crees Thunder gana
  >80% basándote en la temporada, edge = 80-77.5 = 2.5 pts con
  sizing mínimo. Si crees >82.5%, edge = 5 pts → OPERA con $3.

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
