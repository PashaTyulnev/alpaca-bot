"""Q+V-Strategie (siehe long_only.py) live auf dem Alpaca-Paper-Konto.

Befehle:
  python paper_trade.py plan        Zielauswahl mit aktuellen Daten berechnen und anzeigen – handelt NICHT
  python paper_trade.py tick        Für Cron (alle 15 Min.): entscheidet selbst, was zu tun ist
  python paper_trade.py rebalance   Sofort umschichten (nur bei offener Börse)
  python paper_trade.py status      Eigene Positionen und Vergleich mit SPY

Ablauf von `tick` (die Zeitzone des Servers spielt keine Rolle, maßgeblich ist die Börsenuhr von Alpaca):
  * Monat noch nicht umgeschichtet und Börse öffnet heute: vorbörslich die Zielauswahl berechnen.
  * Börse seit mindestens 15 Minuten offen und Monat noch nicht umgeschichtet: umschichten.
    (Normalerweise am ersten Handelstag des Monats; beim allerersten Start sofort am nächsten Handelstag.)
  * Nach Börsenschluss: Depotwert und SPY-Schlusskurs einmal pro Handelstag in data/live/performance.csv.

Sicherheit: Das Skript verwaltet nur Stückzahlen, die es selbst gekauft hat (Buchhaltung in data/live/state.json),
und nur das ihm zugeordnete Bargeld. Andere Positionen im Konto werden weder verkauft noch eingerechnet.
"""

import argparse
import json
import sys
import time
import gzip
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import backtest as bt
from alpaca_download import NY, Alpaca, load_env
from download_data import PROJECT_DIR, load_config, load_nasdaq_common, resolve_data_dir
from fundamentals_test import factor_panel
from long_only import MAIN, QV, composite, top_ew
from sec_fundamentals import build_annual, build_shares, fact_rows, facts_frame

MARKET = "SPY"
TRADE_DELAY_MIN = 15  # nicht in den ersten, besonders unruhigen Minuten handeln
CASH_BUFFER = 0.02  # 2 % Reserve für Kursbewegungen zwischen Planung und Ausführung
MIN_TRADE_USD = 10.0
SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def to_alpaca(ticker: str) -> str:
    return ticker.replace("-", ".")


def to_yahoo(symbol: str) -> str:
    return symbol.replace(".", "-")


# ---------- Zustand ----------

class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def __getitem__(self, k):
        return self.data.get(k)

    def __setitem__(self, k, v):
        self.data[k] = v

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# ---------- Zielauswahl ----------

def sec_get(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"SEC nicht erreichbar: {url}")


def compute_plan(api: Alpaca, env: dict, data_dir: Path) -> pd.DataFrame:
    """Berechnet die Q+V-Auswahl mit denselben Funktionen wie der Backtest, Daten bis zum letzten Börsenschluss."""
    ua = env.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("SEC_USER_AGENT fehlt in .env (Pflicht laut SEC: Name + E-Mail).")
    today = pd.Timestamp.now(tz=NY).normalize().tz_localize(None)

    candidates = load_nasdaq_common(data_dir)
    assets = {a["symbol"]: a for a in api.get(api.trading_url + "/assets", {"status": "active", "asset_class": "us_equity"})}
    symbols = [to_alpaca(t) for t in candidates if to_alpaca(t) in assets and assets[to_alpaca(t)]["tradable"]]
    log(f"Kurse laden: {len(symbols)} handelbare Nasdaq-Stammaktien ...")
    # Ende als genauer Zeitpunkt: Gesamtmarktdaten gibt der kostenlose Zugang erst nach 15 Minuten frei
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = api.daily_bars(symbols, f"{today - pd.Timedelta(days=420):%Y-%m-%d}", end)
    bars = bars[bars["date"] < today]  # nur abgeschlossene Handelstage
    bars["ticker"] = bars["symbol"].map(to_yahoo)
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = bars.pivot(index="date", columns="ticker", values="volume").reindex_like(close)
    panel = bt.Panel(close=close, open=close, raw_close=close, volume=volume)
    asof = close.index[-1]
    universe = bt.liquid_universe(panel, MAIN["top"]).iloc[[-1]]
    liquid = universe.columns[universe.iloc[0].fillna(False).to_numpy()]
    log(f"Stand {asof:%d.%m.%Y}: {len(liquid)} liquideste Aktien im Universum")

    sec_map = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in json.loads(sec_get(SEC_TICKERS, ua)).values()}
    tickers = pd.DataFrame([(t, sec_map[t]) for t in liquid if t in sec_map], columns=["ticker", "cik"])
    log(f"Fundamentaldaten von der SEC laden: {tickers['cik'].nunique()} Firmen ...")
    rows = []
    for i, cik in enumerate(sorted(tickers["cik"].unique()), 1):
        try:
            rows += fact_rows(cik, json.loads(sec_get(SEC_COMPANYFACTS.format(cik=cik), ua)))
        except urllib.error.HTTPError:
            pass  # keine XBRL-Daten (z. B. ausländische Firma)
        time.sleep(0.12)  # SEC erlaubt höchstens 10 Anfragen pro Sekunde
        if i % 250 == 0:
            log(f"  {i} Firmen gelesen")
    facts = facts_frame(rows)
    annual, shares = build_annual(facts), build_shares(facts)

    sp = api.splits(f"{today - pd.Timedelta(days=450):%Y-%m-%d}", f"{today:%Y-%m-%d}")
    splits = pd.DataFrame({"ticker": sp["symbol"].map(to_yahoo), "date": sp["date"], "stock_splits": sp["ratio"]})
    for df, cols in ((annual, ["end", "available"]), (shares, ["shares_date", "filed"]), (splits, ["date"])):
        df[cols] = df[cols].astype("datetime64[ns]")

    month_end = pd.DatetimeIndex([asof]).astype("datetime64[ns]")
    fp = factor_panel(SimpleNamespace(raw_close=close.reindex(month_end)), annual, shares, tickers, splits, month_end)
    eligible = universe.set_axis(month_end) & fp["mcap"].unstack("ticker").reindex(index=month_end, columns=universe.columns).notna()
    score = composite(fp, QV, eligible, 3)
    weights = top_ew(score, frac=MAIN["frac"]).iloc[0]
    picked = weights[weights > 0].index
    plan = pd.DataFrame({"ticker": picked, "symbol": [to_alpaca(t) for t in picked],
                         "score": score.iloc[0][picked].to_numpy(), "weight": weights[picked].to_numpy()})
    plan = plan.sort_values("score", ascending=False, ignore_index=True)
    log(f"Zielauswahl: {len(plan)} Aktien aus {int(eligible.iloc[0].sum())} mit Börsenwert (Score aus {len(QV)} Kennzahlen)")
    return plan


# ---------- Handel ----------

def latest_prices(api: Alpaca, symbols: list[str]) -> dict[str, float]:
    prices = {}
    for i in range(0, len(symbols), 100):
        # Kostenloser Zugang: Echtzeitkurse nur von IEX (für die Ordergröße genau genug)
        snap = api.get(api.data_url + "/stocks/snapshots", {"symbols": ",".join(symbols[i:i + 100]), "feed": "iex"}) or {}
        for sym, s in snap.items():
            trade = (s or {}).get("latestTrade") or {}
            if trade.get("p"):
                prices[sym] = float(trade["p"])
    return prices


def wait_for(api: Alpaca, orders: list[dict], timeout: int = 180) -> list[dict]:
    """Wartet, bis alle Orders abgeschlossen sind (ausgeführt, storniert oder abgelehnt)."""
    done, pending = [], list(orders)
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        time.sleep(2)
        still = []
        for o in pending:
            cur = api.get(f"{api.trading_url}/orders/{o['id']}", {})
            (done if cur["status"] in ("filled", "canceled", "expired", "rejected") else still).append(cur)
        pending = still
    for o in pending:  # nach Ablauf der Frist: Rest stornieren, Teilausführung zählt
        try:
            api.request("DELETE", f"{api.trading_url}/orders/{o['id']}")
        except RuntimeError:
            pass  # inzwischen ausgeführt
        done.append(api.get(f"{api.trading_url}/orders/{o['id']}", {}))
    return done


def submit(api: Alpaca, symbol: str, side: str, tag: str, qty: float | None = None, notional: float | None = None) -> dict | None:
    # Eindeutige Kennung je Order: Wiederholung nach Netzwerkfehler wird von Alpaca als Duplikat abgelehnt
    payload = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day",
               "client_order_id": f"{tag}-{int(time.time())}"}
    if notional is not None:
        payload["notional"] = f"{notional:.2f}"
    else:
        payload["qty"] = f"{qty:.9f}".rstrip("0").rstrip(".")
    try:
        return api.request("POST", api.trading_url + "/orders", payload=payload)
    except RuntimeError as e:
        log(f"  Order abgelehnt {side} {symbol}: {e}")
        return None


def book(state: State, filled: list[dict]) -> None:
    """Eigene Buchhaltung: Stückzahlen und Bargeld der Strategie nach Ausführungen anpassen."""
    holdings, cash = state["holdings"], state["cash"]
    for o in filled:
        q = float(o.get("filled_qty") or 0)
        if q <= 0:
            continue
        px = float(o["filled_avg_price"])
        sign = 1 if o["side"] == "buy" else -1
        holdings[o["symbol"]] = holdings.get(o["symbol"], 0.0) + sign * q
        cash -= sign * q * px
        if abs(holdings[o["symbol"]]) < 1e-9:
            del holdings[o["symbol"]]
    state["holdings"], state["cash"] = holdings, cash


def rebalance(api: Alpaca, env: dict, data_dir: Path, state: State, plan: pd.DataFrame) -> None:
    month = pd.Timestamp.now(tz=NY).strftime("%Y-%m")
    acct = api.get(api.trading_url + "/account", {})
    positions = {p["symbol"]: p for p in api.get(api.trading_url + "/positions", {})}
    assets = {a["symbol"]: a for a in api.get(api.trading_url + "/assets", {"status": "active", "asset_class": "us_equity"})}

    if state["holdings"] is None:  # erster Start: Budget = Depotwert ohne fremde Positionen
        foreign = sum(float(p["market_value"]) for p in positions.values())
        state["holdings"], state["cash"] = {}, float(acct["equity"]) - foreign
        state["start"] = {"date": f"{pd.Timestamp.now(tz=NY):%Y-%m-%d}", "budget": state["cash"]}
        log(f"Erster Start: Budget der Strategie {state['cash']:,.2f} $ (Depotwert ohne die {len(positions)} vorhandenen Positionen)")

    holdings = state["holdings"]
    symbols = sorted(set(plan["symbol"]) | set(holdings))
    prices = latest_prices(api, symbols)
    nav = state["cash"] + sum(q * prices.get(s, float(positions.get(s, {}).get("current_price", 0))) for s, q in holdings.items())
    target_value = nav * (1 - CASH_BUFFER) / len(plan)
    log(f"Umschichten {month}: Wert der Strategie {nav:,.2f} $, {len(plan)} Ziele à {target_value:,.2f} $")

    # 1. Verkaufen: nicht mehr ausgewählte und übergewichtete Positionen
    sells = []
    for sym, q in holdings.items():
        px = prices.get(sym)
        if not px:
            log(f"  Kein Kurs für {sym}, Verkauf übersprungen")
            continue
        excess = q * px - (target_value if sym in set(plan["symbol"]) else 0.0)
        if excess > max(MIN_TRADE_USD, 0.01 * target_value):
            sell_q = q if sym not in set(plan["symbol"]) else excess / px
            if not assets.get(sym, {}).get("fractionable", False):
                sell_q = np.floor(sell_q) if sym in set(plan["symbol"]) else q
            if sell_q > 0:
                o = submit(api, sym, "sell", f"qv-{month}-s-{sym}", qty=sell_q)
                sells += [o] if o else []
    if sells:
        log(f"  {len(sells)} Verkäufe gesendet, warte auf Ausführung ...")
        book(state, wait_for(api, sells))
        state.save()

    # 2. Kaufen: untergewichtete und neue Positionen
    buys, skipped = [], []
    for sym in plan["symbol"]:
        px = prices.get(sym)
        a = assets.get(sym, {})
        if not px or not a.get("tradable"):
            skipped.append(f"{sym} (kein Kurs/nicht handelbar)")
            continue
        missing = target_value - holdings.get(sym, 0.0) * px
        if missing <= max(MIN_TRADE_USD, 0.01 * target_value):
            continue
        if a.get("fractionable"):
            o = submit(api, sym, "buy", f"qv-{month}-b-{sym}", notional=min(missing, state["cash"] * 0.995))
        elif np.floor(missing / px) >= 1:
            o = submit(api, sym, "buy", f"qv-{month}-b-{sym}", qty=float(np.floor(missing / px)))
        else:
            skipped.append(f"{sym} (eine Aktie kostet {px:,.0f} $, Ziel {missing:,.0f} $, nicht fraktionierbar)")
            continue
        buys += [o] if o else []
    if buys:
        log(f"  {len(buys)} Käufe gesendet, warte auf Ausführung ...")
        book(state, wait_for(api, buys))
    for s in skipped:
        log(f"  Übersprungen: {s}")

    state["last_rebalance"] = month
    state.save()
    plan.to_csv(data_dir / "live" / f"plan_{month}.csv", index=False)
    log(f"Fertig: {len(state['holdings'])} Positionen, Bargeld der Strategie {state['cash']:,.2f} $")


# ---------- Protokoll ----------

def snapshot(api: Alpaca, data_dir: Path, state: State, day: str) -> None:
    """Schlusswert der Strategie und SPY-Schlusskurs eines Handelstags protokollieren."""
    holdings = state["holdings"] or {}
    symbols = sorted(set(holdings) | {MARKET})
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = api.daily_bars(symbols, day, end)
    bars = bars[bars["date"] == pd.Timestamp(day)]
    closes = dict(zip(bars["symbol"], bars["close"]))
    if MARKET not in closes:
        log(f"Noch kein SPY-Schlusskurs für {day}, später erneut")
        return
    value = (state["cash"] or 0.0) + sum(q * closes.get(s, np.nan) for s, q in holdings.items())
    path = data_dir / "live" / "performance.csv"
    row = pd.DataFrame([{"date": day, "strategy_value": round(value, 2), "spy_close": closes[MARKET], "positions": len(holdings)}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    state["last_snapshot"] = day
    state.save()
    log(f"Protokoll {day}: Strategie {value:,.2f} $, SPY {closes[MARKET]:.2f}")


def status(data_dir: Path, state: State) -> None:
    print(f"Letzte Umschichtung: {state['last_rebalance'] or '–'} | Positionen: {len(state['holdings'] or {})} | "
          f"Bargeld der Strategie: {state['cash'] or 0:,.2f} $")
    path = data_dir / "live" / "performance.csv"
    if path.exists():
        perf = pd.read_csv(path)
        if len(perf) >= 2:
            s = perf["strategy_value"].iloc[-1] / perf["strategy_value"].iloc[0] - 1
            m = perf["spy_close"].iloc[-1] / perf["spy_close"].iloc[0] - 1
            print(f"Seit {perf['date'].iloc[0]}: Strategie {s:+.2%} | SPY {m:+.2%} (ohne Dividenden) | Differenz {s - m:+.2%}")
        print(perf.tail(10).to_string(index=False))


# ---------- Cron-Logik ----------

def tick(api: Alpaca, env: dict, data_dir: Path, state: State) -> None:
    clock = api.get(api.trading_url + "/clock", {})
    now = pd.Timestamp(clock["timestamp"]).tz_convert(NY)
    month = now.strftime("%Y-%m")
    needs_rebalance = state["last_rebalance"] != month
    cal = api.calendar(f"{now - pd.Timedelta(days=10):%Y-%m-%d}", f"{now:%Y-%m-%d}")
    today = cal[cal["date"] == now.date()]
    plan_path = data_dir / "live" / f"plan_pending_{now:%Y-%m-%d}.csv"

    if clock["is_open"]:
        open_time = today["open"].iloc[0]
        if needs_rebalance and now >= open_time + pd.Timedelta(minutes=TRADE_DELAY_MIN):
            plan = pd.read_csv(plan_path) if plan_path.exists() else compute_plan(api, env, data_dir)
            rebalance(api, env, data_dir, state, plan)
        return

    next_open = pd.Timestamp(clock["next_open"]).tz_convert(NY)
    if needs_rebalance and next_open.date() == now.date() and not plan_path.exists():
        log("Vorbörslich: Zielauswahl für heute berechnen")
        compute_plan(api, env, data_dir).to_csv(plan_path, index=False)
        return
    # Nach Schluss: letzten abgeschlossenen Handelstag protokollieren
    past = cal[cal["close"] <= now]
    if state["holdings"] is not None and len(past):
        last_day = f"{past['date'].iloc[-1]:%Y-%m-%d}"
        if state["last_snapshot"] != last_day and now >= past["close"].iloc[-1] + pd.Timedelta(minutes=20):
            snapshot(api, data_dir, state, last_day)


def main() -> int:
    parser = argparse.ArgumentParser(description="Q+V-Strategie auf dem Alpaca-Paper-Konto.")
    parser.add_argument("command", choices=["plan", "tick", "rebalance", "status"])
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    args = parser.parse_args()

    env = load_env(PROJECT_DIR / ".env")
    if "paper-api" not in env["ALPACA_TRADING_URL"]:
        raise SystemExit("Sicherheitsstopp: ALPACA_TRADING_URL zeigt nicht auf das Paper-Konto (paper-api.alpaca.markets).")
    data_dir = resolve_data_dir(load_config(args.config))
    (data_dir / "live").mkdir(parents=True, exist_ok=True)
    api = Alpaca(env)
    state = State(data_dir / "live" / "state.json")

    if args.command == "plan":
        plan = compute_plan(api, env, data_dir)
        out = data_dir / "live" / f"plan_preview_{datetime.now():%Y-%m-%d}.csv"
        plan.to_csv(out, index=False)
        print(plan.head(25).to_string(index=False))
        print(f"... insgesamt {len(plan)} Aktien, gespeichert in {out}")
    elif args.command == "tick":
        tick(api, env, data_dir, state)
    elif args.command == "rebalance":
        if not api.get(api.trading_url + "/clock", {})["is_open"]:
            raise SystemExit("Börse geschlossen – Umschichten nur bei offener Börse.")
        rebalance(api, env, data_dir, state, compute_plan(api, env, data_dir))
    elif args.command == "status":
        status(data_dir, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
