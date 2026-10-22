"""Lädt Intraday-Kerzen (Gesamtmarkt/SIP) von Alpaca und speichert je Ticker eine Parquet-Datei.

Start:  python alpaca_download.py                       (5-Min-Kerzen, 500 liquideste Aktien + SPY, ab 2016)
        python alpaca_download.py --timeframe 1Min --top 50 --start 2023-01-01
        python alpaca_download.py --tickers AAPL,MSFT

Ablage: data/alpaca/<timeframe>/<TICKER>.parquet, z. B. data/alpaca/5Min/AAPL.parquet
Zugangsdaten kommen aus .env (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY).
Ein erneuter Lauf lädt nur fehlende neuere Kerzen nach.
"""

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from correlations import MARKET, most_liquid
from download_data import PROJECT_DIR, load_config, resolve_data_dir

NY = "America/New_York"
# Kostenloser Zugang: max. 200 Anfragen/Minute, SIP-Daten erst ab 15 Minuten Verzögerung
MIN_SECONDS_BETWEEN_REQUESTS = 0.32
SIP_DELAY = timedelta(minutes=16)


def load_env(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} fehlt. Bitte ALPACA_API_KEY_ID und ALPACA_API_SECRET_KEY eintragen.")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    missing = [k for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") if not env.get(k)]
    if missing:
        raise SystemExit(f"In {path} fehlen: {', '.join(missing)}")
    env.setdefault("ALPACA_DATA_URL", "https://data.alpaca.markets/v2")
    env.setdefault("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets/v2")
    return env


class Alpaca:
    def __init__(self, env: dict):
        self.headers = {
            "APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"],
            "Accept-Encoding": "gzip",  # JSON komprimiert übertragen, spart ca. 90 % Daten
        }
        self.data_url = env["ALPACA_DATA_URL"].rstrip("/")
        self.trading_url = env["ALPACA_TRADING_URL"].rstrip("/")
        self.last_request = 0.0
        self.lock = threading.Lock()  # Ratenbremse gilt für alle parallelen Downloads gemeinsam

    def get(self, url: str, params: dict) -> dict:
        return self.request("GET", url, params)

    def request(self, method: str, url: str, params: dict | None = None, payload: dict | None = None):
        """HTTP-Anfrage mit Ratenbremse und Wiederholung. Orders dürfen wiederholt werden, weil sie eine
        eindeutige client_order_id tragen – Alpaca lehnt Duplikate ab."""
        full = url + ("?" + urllib.parse.urlencode(params) if params else "")
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {**self.headers, **({"Content-Type": "application/json"} if data else {})}
        for attempt in range(6):
            with self.lock:
                wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - self.last_request)
                if wait > 0:
                    time.sleep(wait)
                self.last_request = time.time()
            try:
                req = urllib.request.Request(full, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:200]
                if e.code == 429:
                    print("  Alpaca drosselt (429), warte 30 s ...")
                    time.sleep(30)
                    continue
                if e.code == 401:
                    raise SystemExit(f"Alpaca lehnt die Zugangsdaten ab (HTTP 401): {body}")
                if e.code == 403:
                    raise SystemExit(f"Alpaca verweigert den Zugriff (HTTP 403): {body}")
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  Netzwerkfehler ({e}), neuer Versuch in 10 s ...")
                time.sleep(10)
        raise RuntimeError("Alpaca nach mehreren Versuchen nicht erreichbar.")

    def calendar(self, start: str, end: str) -> pd.DataFrame:
        """Handelstage mit regulären Öffnungszeiten (berücksichtigt verkürzte Tage, z. B. 13:00 Schluss)."""
        days = self.get(self.trading_url + "/calendar", {"start": start, "end": end})
        cal = pd.DataFrame(days)[["date", "open", "close"]]
        cal["open"] = pd.to_datetime(cal["date"] + " " + cal["open"]).dt.tz_localize(NY)
        cal["close"] = pd.to_datetime(cal["date"] + " " + cal["close"]).dt.tz_localize(NY)
        cal["date"] = pd.to_datetime(cal["date"]).dt.date
        return cal

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "symbols": symbol, "timeframe": timeframe, "feed": "sip", "adjustment": "split", "limit": 10000,
            "start": start.isoformat().replace("+00:00", "Z"), "end": end.isoformat().replace("+00:00", "Z"),
        }
        rows = []
        while True:
            page = self.get(self.data_url + "/stocks/bars", params)
            rows += (page.get("bars") or {}).get(symbol, [])
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).rename(columns={
            "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trades", "vw": "vwap",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        return df[["timestamp", "open", "high", "low", "close", "volume", "trades", "vwap"]]

    def daily_bars(self, symbols: list[str], start: str, end: str, chunk: int = 100) -> pd.DataFrame:
        """Tageskerzen (split-bereinigt) für viele Symbole: lange Tabelle symbol, date, close, volume."""
        rows = []
        for i in range(0, len(symbols), chunk):
            params = {"symbols": ",".join(symbols[i:i + chunk]), "timeframe": "1Day", "feed": "sip",
                      "adjustment": "split", "limit": 10000, "start": start, "end": end}
            while True:
                page = self.get(self.data_url + "/stocks/bars", params)
                for sym, bars in (page.get("bars") or {}).items():
                    rows += [(sym, b["t"], b["c"], b["v"]) for b in bars]
                token = page.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        df = pd.DataFrame(rows, columns=["symbol", "date", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
        return df

    def splits(self, start: str, end: str) -> pd.DataFrame:
        """Alle Aktiensplits (vorwärts und rückwärts) im Zeitraum: symbol, date, ratio (neu/alt)."""
        params = {"types": "forward_split,reverse_split", "start": start, "end": end, "limit": 1000}
        rows = []
        while True:
            page = self.get("https://data.alpaca.markets/v1/corporate-actions", params)
            for kind in ("forward_splits", "reverse_splits"):
                for s in (page.get("corporate_actions") or {}).get(kind, []):
                    rows.append((s["symbol"], s["ex_date"], float(s["new_rate"]) / float(s["old_rate"])))
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        df = pd.DataFrame(rows, columns=["symbol", "date", "ratio"])
        df["date"] = pd.to_datetime(df["date"])
        return df


def add_session(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Markiert jede Kerze als 'pre', 'regular' oder 'post' (bzw. 'closed' an Nicht-Handelstagen)."""
    day = df["timestamp"].dt.date
    c = cal.set_index("date")
    open_, close = day.map(c["open"]), day.map(c["close"])
    df["session"] = "closed"
    trading = open_.notna()
    df.loc[trading & (df["timestamp"] < open_), "session"] = "pre"
    df.loc[trading & (df["timestamp"] >= open_) & (df["timestamp"] < close), "session"] = "regular"
    df.loc[trading & (df["timestamp"] >= close), "session"] = "post"
    return df


def save(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old[~old["timestamp"].isin(df["timestamp"])], df], ignore_index=True)
    df = df.sort_values("timestamp", ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Intraday-Kerzen von Alpaca (SIP) als Parquet speichern.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--timeframe", default="5Min", help="z. B. 1Min, 5Min, 15Min (Standard 5Min)")
    parser.add_argument("--start", default="2016-01-01", help="Startdatum JJJJ-MM-TT (Standard 2016-01-01)")
    parser.add_argument("--top", type=int, default=500, help="Anzahl der liquidesten Aktien (Standard 500)")
    parser.add_argument("--tickers", default="", help="Kommagetrennte Ticker statt der liquidesten")
    parser.add_argument("--workers", type=int, default=6, help="Parallele Downloads (Standard 6)")
    args = parser.parse_args()

    data_dir = resolve_data_dir(load_config(args.config))
    api = Alpaca(load_env(PROJECT_DIR / ".env"))
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or most_liquid(data_dir, args.top)
    if MARKET not in tickers:
        tickers.insert(0, MARKET)  # zuerst laden: wird als Marktvergleich für alle Analysen gebraucht
    # Alpaca schreibt Aktienklassen mit Punkt (BRK.B), Yahoo mit Bindestrich (BRK-B)
    symbol = {t: t.replace("-", ".") for t in tickers}

    end = datetime.now(timezone.utc) - SIP_DELAY
    start_default = pd.Timestamp(args.start, tz=NY).tz_convert("UTC").to_pydatetime()
    cal = api.calendar(args.start, f"{end:%Y-%m-%d}")
    out_dir = data_dir / "alpaca" / args.timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    # Börsenkalender für spätere Analysen mitspeichern
    cal.to_csv(data_dir / "alpaca" / "calendar.csv", index=False)

    print(f"{len(tickers)} Ticker, {args.timeframe}, {args.start} bis {end:%Y-%m-%d %H:%M} UTC | Ziel: {out_dir}")
    failed, loaded = [], []

    def process(n: int, ticker: str) -> None:
        path = out_dir / f"{ticker}.parquet"
        start = start_default
        if path.exists():
            # Nur neuere Kerzen nachladen
            last = pd.read_parquet(path, columns=["timestamp"])["timestamp"].max()
            start = last.tz_convert("UTC").to_pydatetime() + timedelta(seconds=1)
        if start >= end:
            return
        t0 = time.time()
        try:
            df = api.bars(symbol[ticker], args.timeframe, start, end)
            if df.empty and not path.exists():
                raise RuntimeError("Keine Kerzen geliefert (Symbol bei Alpaca unbekannt?).")
            if df.empty:
                print(f"[{n}/{len(tickers)}] {ticker}: keine neuen Kerzen", flush=True)
                return
            df.insert(0, "ticker", ticker)
            df = save(add_session(df, cal), path)
        except Exception as e:
            print(f"[{n}/{len(tickers)}] {ticker}: FEHLER {e}", flush=True)
            failed.append(ticker)
            return
        loaded.append(ticker)
        print(f"[{n}/{len(tickers)}] {ticker}: {len(df):,} Kerzen gesamt, {df['timestamp'].iloc[0]:%Y-%m-%d} bis "
              f"{df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} ({time.time() - t0:.0f} s)".replace(",", "."), flush=True)

    # Parallel laden; die Ratenbremse in Alpaca.get hält das Limit von 200 Anfragen/Minute ein
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(process, n, t) for n, t in enumerate(tickers, 1)]:
            future.result()  # SystemExit (z. B. ungültige Zugangsdaten) weiterreichen

    print(f"\nZusammenfassung: {len(loaded)} aktualisiert, {len(failed)} fehlgeschlagen" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
