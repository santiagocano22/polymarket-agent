"""Autonomous trading loop: runs every LOOP_INTERVAL_SECONDS while active."""
from __future__ import annotations

import asyncio
import html
import logging
import math
from typing import Awaitable, Callable

from .config import Config
from .db import Database
from .llm_client import Decision, LLMClient, LLMResult
from .polymarket_client import Market, PolymarketClient, Position

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]


class Trader:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        poly: PolymarketClient,
        llm: LLMClient,
        notify: Notify,
    ):
        self.cfg = cfg
        self.db = db
        self.poly = poly
        self.llm = llm
        self.notify = notify
        self._stop = asyncio.Event()
        self._trigger = asyncio.Event()  # wake-up early (e.g. on /iniciar)

    def stop(self) -> None:
        self._stop.set()
        self._trigger.set()  # unblock the wait immediately

    def trigger_now(self) -> None:
        """Run the next cycle immediately without waiting for the interval."""
        self._trigger.set()

    async def run(self) -> None:
        log.info(
            "Trader loop started (interval=%ss, dry_run=%s)",
            self.cfg.loop_interval_seconds,
            self.cfg.dry_run,
        )
        while not self._stop.is_set():
            # Clear trigger at the top so any set() during the cycle is preserved.
            self._trigger.clear()
            try:
                if await self.db.is_active():
                    await self._cycle()
            except Exception:
                log.exception("cycle error")
                try:
                    await self.notify("⚠️ Error en ciclo del trader (ver logs).")
                except Exception:
                    pass
            if self._stop.is_set():
                break
            # Wait for interval — but wake up early if trigger_now() is called.
            try:
                await asyncio.wait_for(
                    self._trigger.wait(), timeout=self.cfg.loop_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
        log.info("Trader loop stopped")

    # --------------------------------------------------------------- internal
    async def _cycle(self) -> None:
        strategy = await self.db.get_strategy()
        if not strategy.strip():
            return

        markets, positions, balance = await asyncio.gather(
            self.poly.search_markets(
                limit=60,
                min_volume_24h=self.cfg.market_min_volume_24h,
                max_volume_24h=self.cfg.market_max_volume_24h,
                max_days_to_resolution=self.cfg.market_max_days_to_resolution,
            ),
            self.poly.get_positions(),
            self.poly.get_usdc_balance(),
        )
        markets = markets[: self.cfg.max_markets_per_cycle]

        max_exposure_pct = await self.db.get_max_exposure_pct()
        max_per_trade_usd = await self.db.get_max_per_trade_usd()
        stop_loss_usd     = await self.db.get_stop_loss_usd()
        max_open_positions = await self.db.get_max_open_positions()
        initial_bankroll  = await self.db.get_initial_bankroll()

        positions_value = sum(p.current_value_usdc for p in positions)
        total_value     = balance + positions_value
        max_exposure_usd = total_value * (max_exposure_pct / 100.0)

        # ── Sizing dinámico: 10% del bankroll actual si no hay cap explícito ──
        dynamic_max_per_trade = max_per_trade_usd or round(total_value * 0.10, 2)

        # ── Stop-loss absoluto ────────────────────────────────────────────────
        if stop_loss_usd is not None and total_value < stop_loss_usd:
            await self.db.set_active(False)
            await self.notify(
                f"🛑 <b>STOP-LOSS activado</b>\n"
                f"Portafolio: <b>${total_value:.2f}</b> cayó bajo ${stop_loss_usd:.2f}.\n"
                f"Agente pausado. Revisá con /posiciones."
            )
            log.warning("stop-loss triggered: total=%.2f < limit=%.2f", total_value, stop_loss_usd)
            return

        # ── Alerta –30% del bankroll inicial ─────────────────────────────────
        if initial_bankroll and total_value < initial_bankroll * 0.70:
            await self.notify(
                f"⚠️ <b>Alerta: –30% del bankroll inicial</b>\n"
                f"Inicial: ${initial_bankroll:.2f} → Actual: ${total_value:.2f} "
                f"({((total_value/initial_bankroll)-1)*100:.1f}%)\n"
                f"Agente continúa pero revisá tu estrategia."
            )

        log.info(
            "cycle: balance=%.2f pos_value=%.2f total=%.2f "
            "max_exposure=%.2f max_per_trade=%.2f open=%d/%s markets=%d",
            balance, positions_value, total_value, max_exposure_usd,
            dynamic_max_per_trade, len(positions),
            str(max_open_positions) if max_open_positions else "∞",
            len(markets),
        )

        # ── Guard: skip LLM if there's nothing actionable ────────────────────────
        # Cheapest possible BUY = 5 shares × $0.05 = $0.25.  If we have no
        # balance AND no open positions worth monitoring, the LLM call wastes
        # money without producing any executable decisions.
        MIN_BUY_BUDGET = 0.25
        if balance < MIN_BUY_BUDGET and not positions:
            log.info("cycle: skipping LLM — balance $%.2f, no positions", balance)
            return

        result = await self.llm.decide(
            strategy=strategy,
            markets=markets,
            positions=positions,
            usdc_balance=balance,
            max_exposure_usd=max_exposure_usd,
            max_per_trade_usd=dynamic_max_per_trade,
        )

        actionable = [d for d in result.decisions if d.action in ("BUY", "SELL")]
        if not actionable:
            log.info("LLM produced no actionable decisions — analysis: %s", result.analysis)
            return

        for d in actionable:
            await self._execute(
                d,
                positions=positions,
                markets=markets,
                balance=balance,
                positions_value=positions_value,
                max_exposure_usd=max_exposure_usd,
                max_per_trade_usd=dynamic_max_per_trade,
                max_open_positions=max_open_positions,
            )

    async def _execute(
        self,
        d: Decision,
        *,
        positions: list[Position],
        markets: list[Market],
        balance: float,
        positions_value: float,
        max_exposure_usd: float,
        max_per_trade_usd: float | None,
        max_open_positions: int | None,
    ) -> None:
        # Enforce CLOB minimum: 5 shares × limit_price (dynamic, not a fixed dollar amount)
        if d.action == "BUY" and d.limit_price > 0:
            min_cost = round(5.0 * d.limit_price, 2)
            if d.size_usdc < min_cost:
                log.info("bumping size_usdc %.2f → %.2f (5 shares × %.3f)", d.size_usdc, min_cost, d.limit_price)
                d.size_usdc = min_cost

        reject = _risk_check(
            d,
            balance=balance,
            positions_value=positions_value,
            max_exposure_usd=max_exposure_usd,
            max_per_trade_usd=max_per_trade_usd,
            positions=positions,
            max_open_positions=max_open_positions,
        )
        if reject:
            log.warning("risk rejected: %s — %s", d, reject)
            await self._log_and_notify(
                d, status=f"REJECTED:{reject}", response=None
            )
            return

        market_title = d.market_title or _lookup_title(d.market_id, markets)

        if self.cfg.dry_run:
            await self._log_and_notify(
                d, status="DRY_RUN", market_title=market_title, response=None
            )
            return

        try:
            if d.action == "BUY":
                if d.limit_price <= 0:
                    await self._log_and_notify(
                        d, status="REJECTED:limit_price_missing",
                        market_title=market_title, response=None,
                    )
                    return
                # Re-fetch balance justo antes de comprar (puede haber bajado en este ciclo)
                fresh_balance = await self.poly.get_usdc_balance()
                min_cost = round(5.0 * d.limit_price, 2)
                if fresh_balance < min_cost:
                    await self._log_and_notify(
                        d,
                        status=f"SKIPPED:balance ${fresh_balance:.2f} < min ${min_cost:.2f}",
                        market_title=market_title, response=None,
                    )
                    return
                # Usar el menor entre lo pedido y el balance real disponible
                # Floor to 2 decimals (never round up) so the CLOB order
                # amount never exceeds the actual available balance.
                # e.g. $12.0093 → $12.00, not $12.01
                safe_balance = math.floor(fresh_balance * 100) / 100
                d.size_usdc = min(d.size_usdc, safe_balance)
                resp = await self.poly.buy_limit(d.token_id, d.limit_price, d.size_usdc)
            else:
                # No intentar vender posiciones casi sin valor — el CLOB requiere
                # USDC para vender a precios muy bajos (vender YES a 0.001 = comprar NO a 0.999)
                MIN_SELL_VALUE = 0.50
                held = next((p for p in positions if p.token_id == d.token_id), None)
                if held and held.current_value_usdc < MIN_SELL_VALUE:
                    await self._log_and_notify(
                        d,
                        status=f"SKIPPED:position_value ${held.current_value_usdc:.3f} < ${MIN_SELL_VALUE} (let resolve)",
                        market_title=market_title, response=None,
                    )
                    return
                shares = _size_to_shares_for_sell(d, positions)
                if shares <= 0:
                    await self._log_and_notify(
                        d, status="REJECTED:no-shares-for-sell",
                        market_title=market_title, response=None,
                    )
                    return
                sell_price = max(0.001, min(0.999, d.limit_price)) if d.limit_price > 0 else 0.95
                resp = await self.poly.sell_limit(d.token_id, sell_price, shares)
            status = "OK" if _order_ok(resp) else f"FAIL:{_order_err(resp)}"
            await self._log_and_notify(
                d, status=status, market_title=market_title, response=resp
            )
        except Exception as e:
            log.exception("execute error")
            await self._log_and_notify(
                d, status=f"ERROR:{e}", market_title=market_title, response=None
            )

    async def _log_and_notify(
        self,
        d: Decision,
        *,
        status: str,
        market_title: str | None = None,
        response: dict | None = None,
    ) -> None:
        title = market_title or d.market_title
        price = None
        if response and isinstance(response, dict):
            try:
                price = float(response.get("avgPrice") or response.get("price") or 0) or None
            except Exception:
                price = None
        await self.db.log_trade(
            market_id=d.market_id,
            market_title=title,
            token_id=d.token_id,
            side=d.action,
            size_usdc=d.size_usdc,
            price=price,
            status=status,
            dry_run=self.cfg.dry_run,
            reasoning=d.reasoning,
            response=response,
        )
        icon = "🧪" if status == "DRY_RUN" else ("✅" if status == "OK" else "⚠️")
        e = html.escape  # escape < > & " in all dynamic content
        price_str = f" @ {d.limit_price:.3f}" if d.limit_price > 0 else ""
        edge_str  = f"  edge={d.edge*100:.1f}%" if d.edge else ""
        msg = (
            f"{icon} <b>{d.action}</b> ${d.size_usdc:.2f}{e(price_str)}{e(edge_str)}\n"
            f"<i>{e(title or '')}</i>\n"
            f"Status: <code>{e(status)}</code>\n"
            f"Razón: {e(d.reasoning)}"
        )
        try:
            await self.notify(msg)
        except Exception:
            log.exception("notify failed")


def _risk_check(
    d: Decision,
    *,
    balance: float,
    positions_value: float,
    max_exposure_usd: float,
    max_per_trade_usd: float | None,
    positions: list[Position],
    max_open_positions: int | None,
) -> str | None:
    if d.size_usdc <= 0:
        return "size<=0"
    # max_per_trade solo aplica a BUY (limitar riesgo de entrada), nunca a SELL
    if d.action == "BUY" and max_per_trade_usd is not None and d.size_usdc > max_per_trade_usd:
        return f"exceeds max_per_trade(${max_per_trade_usd:.2f})"
    if d.action == "BUY":
        if d.size_usdc > balance:
            return "insufficient_usdc"
        projected_exposure = positions_value + d.size_usdc
        if projected_exposure > max_exposure_usd:
            return f"exceeds max_exposure(${max_exposure_usd:.2f})"
        if max_open_positions is not None and len(positions) >= max_open_positions:
            return f"max_open_positions({max_open_positions}) reached"
    if d.action == "SELL":
        held = next((p for p in positions if p.token_id == d.token_id), None)
        if held is None:
            return "no_position_for_token"
    return None


def _size_to_shares_for_sell(d: Decision, positions: list[Position]) -> float:
    held = next((p for p in positions if p.token_id == d.token_id), None)
    if held is None or held.size <= 0:
        return 0.0
    # If the LLM asks to liquidate more USDC than held → sell all.
    if held.current_value_usdc <= 0 or d.size_usdc >= held.current_value_usdc:
        return held.size
    ratio = d.size_usdc / held.current_value_usdc
    return held.size * ratio


def _lookup_title(market_id: str, markets: list[Market]) -> str:
    m = next((m for m in markets if m.id == market_id), None)
    return m.question if m else ""


def _order_ok(resp: dict | None) -> bool:
    if not isinstance(resp, dict):
        return False
    return bool(resp.get("success")) or resp.get("status") in ("matched", "live")


def _order_err(resp: dict | None) -> str:
    if not isinstance(resp, dict):
        return "unknown"
    return str(resp.get("errorMsg") or resp.get("error") or resp)[:120]


async def sell_all_positions(
    poly: PolymarketClient, db: Database, notify: Notify, dry_run: bool
) -> None:
    positions = await poly.get_positions()
    if not positions:
        await notify("No hay posiciones abiertas para liquidar.")
        return
    await notify(f"Liquidando {len(positions)} posiciones…")
    for p in positions:
        try:
            if dry_run:
                status, resp = "DRY_RUN", None
            else:
                resp = await poly.sell_limit(p.token_id, 0.99, p.size)
                status = "OK" if _order_ok(resp) else f"FAIL:{_order_err(resp)}"
        except Exception as e:
            status, resp = f"ERROR:{e}", None
        await db.log_trade(
            market_id=p.market_id,
            market_title=p.title,
            token_id=p.token_id,
            side="SELL",
            size_usdc=p.current_value_usdc,
            price=None,
            status=status,
            dry_run=dry_run,
            reasoning="manual liquidate-all",
            response=resp,
        )
        icon = "🧪" if status == "DRY_RUN" else ("✅" if status == "OK" else "⚠️")
        await notify(f"{icon} SELL {html.escape(p.title)} ({p.outcome}) — {html.escape(status)}")
