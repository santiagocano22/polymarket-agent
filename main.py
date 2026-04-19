"""Entry point: runs the Telegram bot and the autonomous trader loop together."""
from __future__ import annotations

import asyncio
import logging
import signal

from src.config import load_config
from src.db import Database
from src.llm_client import LLMClient
from src.polymarket_client import PolymarketClient
from src.telegram_bot import build_application, make_notifier
from src.trader import Trader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main() -> None:
    cfg = load_config()
    log = logging.getLogger("main")
    log.info(
        "Config loaded. EOA=%s wallet=%s sig_type=%s dry_run=%s",
        cfg.eoa_address, cfg.polymarket_wallet_address,
        cfg.polymarket_signature_type, cfg.dry_run,
    )

    db = Database(cfg.db_path)
    await db.init()

    poly = PolymarketClient(cfg)
    llm = LLMClient(cfg)

    # trader_ref lets on_activate reference the trader before it's created
    # (the lambda is only called after /iniciar, well after setup is complete).
    trader_ref: list[Trader] = []
    app = build_application(
        cfg, db, poly,
        on_activate=lambda: trader_ref and trader_ref[0].trigger_now(),
    )
    notify = await make_notifier(app, cfg.telegram_chat_id)
    trader = Trader(cfg, db, poly, llm, notify)
    trader_ref.append(trader)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    modo = "SIMULACIÓN" if cfg.dry_run else "REAL"
    await notify(f"🤖 Agente online. Modo: {modo}. Usá /help para ver comandos.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # Windows

    trader_task = asyncio.create_task(trader.run(), name="trader")

    try:
        await stop_event.wait()
    finally:
        log.info("shutting down…")
        trader.stop()
        await asyncio.gather(trader_task, return_exceptions=True)
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await poly.close()


if __name__ == "__main__":
    asyncio.run(main())
