"""Telegram bot: command interface to the agent."""
from __future__ import annotations

import datetime as dt
import html
import logging
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .db import Database
from .polymarket_client import PolymarketClient
from .trader import sell_all_positions

log = logging.getLogger(__name__)


HELP = (
    "<b>Comandos disponibles</b>\n"
    "/estrategia &lt;texto&gt; — define/reemplaza la estrategia\n"
    "/ver_estrategia — muestra la estrategia actual\n"
    "/iniciar — arranca el loop autónomo\n"
    "/pausar — pausa el loop\n"
    "/estado — estado del agente\n"
    "/balance — USDC disponible\n"
    "/posiciones — posiciones abiertas\n"
    "/historial — últimos trades\n"
    "/vender_todo — liquida todas las posiciones\n"
    "/cancelarordenes — cancela todas las órdenes límite abiertas\n"
    "/limites &lt;pct&gt; [max_usd] — ajusta límites de riesgo\n"
    "/ver_limites — muestra límites actuales\n"
    "/dryrun &lt;on|off&gt; — alterna modo simulación\n"
    "/help — esta ayuda"
)


def build_application(
    cfg: Config,
    db: Database,
    poly: PolymarketClient,
    on_activate: "Callable[[], None] | None" = None,
) -> Application:
    app = Application.builder().token(cfg.telegram_bot_token).build()

    def _auth(update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == cfg.telegram_chat_id

    def guarded(handler):
        async def wrapped(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            if not _auth(update):
                log.warning(
                    "unauthorized message from chat_id=%s",
                    update.effective_chat.id if update.effective_chat else "?",
                )
                return
            await handler(update, ctx)
        return wrapped

    # ------------------------------------------------------------- handlers
    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        modo = "SIMULACIÓN" if cfg.dry_run else "REAL"
        await update.message.reply_text(
            f"Hola. Agente listo. Modo: {modo}\n\n{HELP}",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELP, parse_mode=ParseMode.HTML)

    async def cmd_estrategia(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            full = (update.message.text or "").strip()
            # Strip the command prefix (handles /estrategia and /estrategia@botname)
            parts = full.split(None, 1)
            strategy_text = parts[1].strip() if len(parts) > 1 else ""
            if not strategy_text:
                await update.message.reply_text(
                    "Uso: /estrategia <texto>\n"
                    "O mandá el texto de la estrategia como mensaje normal (sin comando)."
                )
                return
            await db.set_strategy(strategy_text)
            await update.message.reply_text(f"Estrategia guardada ({len(strategy_text)} chars).")
        except Exception as e:
            log.exception("cmd_estrategia error")
            await update.message.reply_text(f"Error guardando estrategia: {e}")

    async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Cualquier mensaje de texto que no sea comando se guarda como estrategia."""
        try:
            text = (update.message.text or "").strip()
            if not text:
                return
            await db.set_strategy(text)
            await update.message.reply_text(
                f"Estrategia guardada ({len(text)} chars).\n"
                "Usá /ver_estrategia para confirmar, /iniciar para arrancar."
            )
        except Exception as e:
            log.exception("handle_text error")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_ver_estrategia(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s = await db.get_strategy()
        # Send as plain text (no parse_mode) to avoid HTML conflicts in strategy text
        await update.message.reply_text(
            f"Estrategia actual:\n\n{s or '(vacía)'}",
            parse_mode=None,
        )

    async def cmd_iniciar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not (await db.get_strategy()).strip():
            await update.message.reply_text(
                "No hay estrategia definida. Mandá el texto de tu estrategia primero."
            )
            return
        # Guardar bankroll inicial la primera vez
        if not await db.get_initial_bankroll():
            try:
                bal = await poly.get_usdc_balance()
                positions = await poly.get_positions()
                total = bal + sum(p.current_value_usdc for p in positions)
                await db.set_initial_bankroll(total)
                await update.message.reply_text(f"Bankroll inicial registrado: ${total:.2f}")
            except Exception:
                pass
        await db.set_active(True)
        if on_activate:
            on_activate()  # wake up the trader loop immediately
        mode = "SIMULACIÓN" if cfg.dry_run else "LIVE"
        await update.message.reply_text(
            f"Agente INICIADO ({mode}). Intervalo: {cfg.loop_interval_seconds}s.\n"
            f"Límites activos: exposición 30%, max/trade 10% bankroll, stop-loss /ver_limites"
        )

    async def cmd_pausar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await db.set_active(False)
        await update.message.reply_text("Agente PAUSADO.")

    async def cmd_estado(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        active = await db.is_active()
        max_exp = await db.get_max_exposure_pct()
        max_trade = await db.get_max_per_trade_usd()
        await update.message.reply_text(
            f"Activo: {'SÍ' if active else 'NO'}\n"
            f"Modo: {'DRY_RUN' if cfg.dry_run else 'LIVE'}\n"
            f"Intervalo: {cfg.loop_interval_seconds}s\n"
            f"Max exposición: {max_exp:.0f}%\n"
            f"Max por trade: "
            f"{('$'+format(max_trade,'.2f')) if max_trade else '(libre)'}"
        )

    async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            bal = await poly.get_usdc_balance()
            await update.message.reply_text(f"USDC disponible: ${bal:.2f}")
        except Exception as e:
            await update.message.reply_text(f"Error leyendo balance: {e}")

    async def cmd_posiciones(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            positions = await poly.get_positions()
        except Exception as e:
            await update.message.reply_text(f"Error leyendo posiciones: {e}")
            return
        if not positions:
            await update.message.reply_text("Sin posiciones abiertas.")
            return
        lines = [f"<b>{len(positions)} posiciones:</b>"]
        total = 0.0
        for p in positions:
            total += p.current_value_usdc
            lines.append(
                f"• {p.title} ({p.outcome}) — {p.size:.2f} @ "
                f"avg ${p.avg_price:.3f} · valor ${p.current_value_usdc:.2f}"
            )
        lines.append(f"\n<b>Valor total:</b> ${total:.2f}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML
        )

    async def cmd_historial(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = await db.recent_trades(10)
        if not trades:
            await update.message.reply_text("Sin trades registrados.")
            return
        lines = ["<b>Últimos trades:</b>"]
        for t in trades:
            ts = dt.datetime.fromtimestamp(t.ts).strftime("%m-%d %H:%M")
            tag = "🧪" if t.dry_run else ""
            lines.append(
                f"<code>{ts}</code> {tag}{t.side} ${(t.size_usdc or 0):.2f} · "
                f"{(t.market_title or '?')[:45]} · {t.status}"
            )
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML
        )

    async def cmd_vender_todo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await db.set_active(False)
        await update.message.reply_text("Pausando y liquidando…")

        async def notify(text: str):
            await ctx.bot.send_message(
                chat_id=cfg.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        await sell_all_positions(poly, db, notify, cfg.dry_run)

    async def cmd_limites(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        parts = (update.message.text or "").split()
        if len(parts) < 2:
            await update.message.reply_text(
                "Uso: /limites <max_exposicion_pct> [max_por_trade_usd]\n"
                "Ejemplo: /limites 40 5"
            )
            return
        try:
            pct = float(parts[1])
            per_trade = float(parts[2]) if len(parts) > 2 else None
        except ValueError:
            await update.message.reply_text("Valores inválidos.")
            return
        await db.set("max_exposure_pct", str(pct))
        await db.set("max_per_trade_usd", str(per_trade) if per_trade else "")
        await update.message.reply_text(
            f"Límites: max_exposición={pct:.0f}% · "
            f"max_por_trade={('$'+format(per_trade,'.2f')) if per_trade else '(libre)'}"
        )

    async def cmd_ver_limites(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pct = await db.get_max_exposure_pct()
        per = await db.get_max_per_trade_usd()
        sl = await db.get_stop_loss_usd()
        maxp = await db.get_max_open_positions()
        await update.message.reply_text(
            f"max_exposición: {pct:.0f}%\n"
            f"max_por_trade: {('$'+format(per,'.2f')) if per else '(libre)'}\n"
            f"stop_loss: {('$'+format(sl,'.2f')) if sl else '(desactivado)'}\n"
            f"max_posiciones: {maxp if maxp else '(sin límite)'}"
        )

    async def cmd_stoplosson(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        parts = (update.message.text or "").split()
        if len(parts) < 2:
            await update.message.reply_text("Uso: /stoplosson <valor_usd>\nEjemplo: /stoplosson 15")
            return
        try:
            val = float(parts[1])
        except ValueError:
            await update.message.reply_text("Valor inválido.")
            return
        await db.set("stop_loss_usd", str(val))
        await update.message.reply_text(f"Stop-loss configurado: si el portafolio cae bajo ${val:.2f}, el agente se pausa automáticamente.")

    async def cmd_stoplossoff(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await db.set("stop_loss_usd", "")
        await update.message.reply_text("Stop-loss desactivado.")

    async def cmd_maxposiciones(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        parts = (update.message.text or "").split()
        if len(parts) < 2:
            await update.message.reply_text("Uso: /maxposiciones <número>\nEjemplo: /maxposiciones 3")
            return
        try:
            val = int(parts[1])
        except ValueError:
            await update.message.reply_text("Valor inválido.")
            return
        await db.set("max_open_positions", str(val))
        await update.message.reply_text(f"Máximo de posiciones abiertas: {val}")

    async def cmd_cancelarordenes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Cancelando todas las órdenes abiertas…")
        try:
            result = await poly.cancel_all_orders()
            orders = await poly.get_open_orders()
            await update.message.reply_text(
                f"✅ Órdenes canceladas.\nRespuesta: {str(result)[:200]}\n"
                f"Órdenes restantes: {len(orders)}"
            )
        except Exception as e:
            await update.message.reply_text(f"Error cancelando órdenes: {e}")

    async def cmd_dryrun(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        parts = (update.message.text or "").split()
        if len(parts) < 2 or parts[1].lower() not in ("on", "off"):
            await update.message.reply_text("Uso: /dryrun <on|off>")
            return
        new_val = "true" if parts[1].lower() == "on" else "false"
        # Persist to hint user; actual cfg requires .env change + restart.
        await db.set("dry_run_hint", new_val)
        await update.message.reply_text(
            f"Guardado hint dry_run={new_val}.\n"
            "Para aplicar cambiá DRY_RUN en .env y reiniciá el proceso."
        )

    async def cmd_unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Comando no reconocido. /help")

    # ------------------------------------------------------------ register
    app.add_handler(CommandHandler("start", guarded(cmd_start)))
    app.add_handler(CommandHandler("help", guarded(cmd_help)))
    app.add_handler(CommandHandler("estrategia", guarded(cmd_estrategia)))
    app.add_handler(CommandHandler("ver_estrategia", guarded(cmd_ver_estrategia)))
    app.add_handler(CommandHandler("iniciar", guarded(cmd_iniciar)))
    app.add_handler(CommandHandler("pausar", guarded(cmd_pausar)))
    app.add_handler(CommandHandler("estado", guarded(cmd_estado)))
    app.add_handler(CommandHandler("balance", guarded(cmd_balance)))
    app.add_handler(CommandHandler("posiciones", guarded(cmd_posiciones)))
    app.add_handler(CommandHandler("historial", guarded(cmd_historial)))
    app.add_handler(CommandHandler("vender_todo", guarded(cmd_vender_todo)))
    app.add_handler(CommandHandler("limites", guarded(cmd_limites)))
    app.add_handler(CommandHandler("ver_limites", guarded(cmd_ver_limites)))
    app.add_handler(CommandHandler("stoplosson", guarded(cmd_stoplosson)))
    app.add_handler(CommandHandler("stoplossoff", guarded(cmd_stoplossoff)))
    app.add_handler(CommandHandler("maxposiciones", guarded(cmd_maxposiciones)))
    app.add_handler(CommandHandler("cancelarordenes", guarded(cmd_cancelarordenes)))
    app.add_handler(CommandHandler("dryrun", guarded(cmd_dryrun)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, guarded(handle_text)))
    app.add_handler(MessageHandler(filters.COMMAND, guarded(cmd_unknown)))

    return app


async def make_notifier(app: Application, chat_id: int):
    async def notify(text: str) -> None:
        try:
            await app.bot.send_message(
                chat_id=chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            log.warning("telegram notify failed: %s", e)
    return notify
