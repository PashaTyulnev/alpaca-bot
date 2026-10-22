"""Lädt historische Kursdaten über yfinance und speichert je Ticker und Intervall eine Parquet-Datei.

Start:  python download_data.py                 (nutzt config.toml)
        python download_data.py --limit 20      (nur die ersten 20 Ticker, zum Testen)
        python download_data.py --config andere_config.toml

Ablage: data/prices/<intervall>/<TICKER>.parquet, z. B. data/prices/1d/AAPL.parquet
"""

import argparse
import csv
import io
import re
import sys
import time
import tomllib
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

PROJECT_DIR = Path(__file__).resolve().parent

# Spalten, ohne die eine Datei für den späteren Backtest nicht brauchbar ist
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
# Spalten, die yfinance normalerweise mitliefert; fehlen sie, gibt es nur eine Warnung
ACTION_COLUMNS = ["Dividends", "Stock Splits"]

# Abrufparameter je Intervall. "merge": neue Daten mit vorhandener Datei zusammenführen,
# weil Yahoo nur ein begrenztes Fenster liefert und die Historie sonst verloren ginge.
INTERVALS = {
    "1d": {"period": "max", "time_column": "date", "merge": False},
    "1h": {"lookback_days": 729, "time_column": "timestamp", "merge": True},
}

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
# Keine Stammaktien: Optionsscheine, Bezugsrechte, Units, Vorzugsaktien, Anleihen
NON_COMMON = re.compile(r"\b(?:warrants?|rights?|units?|preferred|notes due|debentures?)\b", re.IGNORECASE)

# Wartezeiten in Sekunden, wenn Yahoo Anfragen drosselt
RATE_LIMIT_WAITS = [60, 180, 600]


class DataError(Exception):
    """Die Antwort von yfinance ist leer oder unbrauchbar."""


class RateLimited(Exception):
    """Yahoo drosselt dauerhaft – weitere Anfragen sind gerade sinnlos."""


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Konfigurationsdatei nicht gefunden: {path}")
    try:
        with path.open("rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Konfigurationsdatei {path} ist kein gültiges TOML: {e}")

    tickers = config.get("tickers", [])
    if not isinstance(tickers, list) or not all(isinstance(t, str) and t.strip() for t in tickers):
        raise SystemExit(f"In {path} muss 'tickers' eine Liste von Symbolen sein, z. B. tickers = [\"SPY\"].")
    config["tickers"] = [t.strip().upper() for t in tickers]

    config["universe"] = config.get("universe") or ""
    if config["universe"] not in ("", "nasdaq-common"):
        raise SystemExit(f"Unbekanntes Universum '{config['universe']}' in {path}. Erlaubt: \"nasdaq-common\" oder \"\".")
    if not config["universe"] and not config["tickers"]:
        raise SystemExit(f"In {path} ist weder ein 'universe' noch eine 'tickers'-Liste angegeben.")

    intervals = config.get("intervals", ["1d"])
    unknown = [i for i in intervals if i not in INTERVALS]
    if unknown or not intervals:
        raise SystemExit(f"Ungültige Intervalle in {path}: {unknown or 'leer'}. Erlaubt: {', '.join(INTERVALS)}.")
    config["intervals"] = intervals

    config.setdefault("data_dir", "data")
    config.setdefault("pause_seconds", 0.5)
    config.setdefault("skip_if_younger_than_hours", 12)
    return config


def resolve_data_dir(config: dict) -> Path:
    data_dir = Path(config["data_dir"])
    return data_dir if data_dir.is_absolute() else PROJECT_DIR / data_dir


def load_nasdaq_common(data_dir: Path) -> list[str]:
    """Holt die aktuelle Nasdaq-Symbolliste und filtert auf Stammaktien.

    Die gefilterte Liste wird unter data/universe/ gespeichert. Ist nasdaqtrader.com nicht
    erreichbar, wird die zuletzt gespeicherte Liste verwendet.
    """
    saved = data_dir / "universe" / "nasdaq-common.csv"
    try:
        with urllib.request.urlopen(NASDAQ_LISTED_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        if saved.exists():
            print(f"Warnung: Nasdaq-Liste nicht abrufbar ({e}). Verwende gespeicherte Liste von {saved}.")
            return pd.read_csv(saved)["ticker"].tolist()
        raise SystemExit(f"Nasdaq-Symbolliste konnte nicht geladen werden ({e}) und es gibt keine gespeicherte Kopie.")

    lines = [l for l in text.splitlines() if l and not l.startswith("File Creation Time")]
    listed = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)
    common = listed[
        (listed["Test Issue"] == "N")
        & (listed["ETF"] == "N")
        & ~listed["Security Name"].str.contains(NON_COMMON)
    ]
    # Yahoo schreibt Aktienklassen mit Bindestrich (BRK.B -> BRK-B)
    universe = pd.DataFrame({
        "ticker": common["Symbol"].str.replace(".", "-", regex=False),
        "name": common["Security Name"],
    }).sort_values("ticker")
    saved.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(saved, index=False)
    print(f"Nasdaq-Liste: {len(listed)} Symbole, davon {len(universe)} Stammaktien -> {saved}")
    return universe["ticker"].tolist()


def fetch(ticker: str, interval: str) -> pd.DataFrame:
    params = INTERVALS[interval]
    kwargs = {"period": params["period"]} if "period" in params else {
        "start": date.today() - timedelta(days=params["lookback_days"])
    }
    # Fehler von yfinance als Exception erhalten statt nur als Log-Ausgabe
    yf.config.debug.hide_exceptions = False

    for attempt, wait in enumerate([0] + RATE_LIMIT_WAITS):
        if wait:
            print(f"  Yahoo drosselt Anfragen, warte {wait} s (Versuch {attempt + 1}/{len(RATE_LIMIT_WAITS) + 1}) ...")
            time.sleep(wait)
        try:
            df = yf.Ticker(ticker).history(
                interval=interval,
                auto_adjust=False,  # Close unbereinigt lassen und Adj Close separat behalten
                actions=True,  # Dividenden und Splits mitliefern
                **kwargs,
            )
            break
        except YFRateLimitError:
            continue
        except Exception as e:
            if "404" in str(e):
                raise DataError("Symbol bei Yahoo Finance nicht gefunden (HTTP 404).") from e
            raise DataError(f"Download fehlgeschlagen ({type(e).__name__}): {e}") from e
    else:
        raise RateLimited("Yahoo drosselt weiterhin.")

    if df is None or df.empty:
        raise DataError("Leere Antwort von Yahoo Finance (kein Handel im Zeitraum oder Symbol unbekannt).")
    return df


def validate_and_clean(ticker: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"Pflichtspalten fehlen: {', '.join(missing)}. Vorhanden: {', '.join(map(str, df.columns))}"
        )
    for col in ACTION_COLUMNS:
        if col not in df.columns:
            print(f"  Warnung: Spalte '{col}' fehlt in der Antwort, wird mit 0 aufgefüllt.")
            df[col] = 0.0

    index = pd.DatetimeIndex(df.index)
    if interval == "1d":
        # Tageskerzen: Zeitzone und Uhrzeit entfernen, nur das Handelsdatum behalten
        if index.tz is not None:
            index = index.tz_localize(None)
        index = index.normalize()
    # Intraday: Zeitstempel mit Zeitzone der Börse (America/New_York) behalten
    df.index = index

    duplicates = df.index[df.index.duplicated()].unique()
    if len(duplicates) > 0:
        shown = ", ".join(str(d) for d in duplicates[:5])
        more = f" (und {len(duplicates) - 5} weitere)" if len(duplicates) > 5 else ""
        raise DataError(f"{len(duplicates)} doppelte Zeitwerte gefunden: {shown}{more}")

    n_nan = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    if n_nan:
        print(f"  Warnung: {n_nan} Zeilen mit fehlenden Kurswerten (bleiben unverändert gespeichert).")

    # Einheitliche, SQL-freundliche Spaltennamen; weitere Spalten (z. B. 'Capital Gains') bleiben erhalten
    ordered = REQUIRED_COLUMNS + ACTION_COLUMNS
    df = df.sort_index()[ordered + [c for c in df.columns if c not in ordered]]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    time_column = INTERVALS[interval]["time_column"]
    df = df.rename_axis(time_column).reset_index()
    if interval == "1d":
        df[time_column] = df[time_column].dt.date
    df.insert(0, "ticker", ticker)
    return df


def save(df: pd.DataFrame, path: Path, interval: str) -> pd.DataFrame:
    """Schreibt atomar: erst in eine temporäre Datei, dann ersetzen. Eine vorhandene
    Datei bleibt bei einem Fehler unangetastet."""
    time_column = INTERVALS[interval]["time_column"]
    if INTERVALS[interval]["merge"] and path.exists():
        old = pd.read_parquet(path)
        # Ältere Zeilen behalten, die Yahoo nicht mehr liefert; überlappende durch neue ersetzen
        old = old[~old[time_column].isin(df[time_column])]
        df = pd.concat([old, df], ignore_index=True).sort_values(time_column, ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return df


def is_fresh(path: Path, max_age_hours: float) -> bool:
    if max_age_hours <= 0 or not path.exists():
        return False
    return time.time() - path.stat().st_mtime < max_age_hours * 3600


def append_log(log_path: Path, rows: list[dict]) -> None:
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "ticker", "interval", "status", "rows", "message"])
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Historische Kursdaten per yfinance als Parquet speichern.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml", help="Pfad zur TOML-Konfiguration")
    parser.add_argument("--limit", type=int, default=0, help="Nur die ersten N Ticker verarbeiten (zum Testen)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(config["tickers"])
    if config["universe"] == "nasdaq-common":
        tickers += [t for t in load_nasdaq_common(data_dir) if t not in tickers]
    if args.limit > 0:
        tickers = tickers[: args.limit]

    run = datetime.now().isoformat(timespec="seconds")
    print(f"{len(tickers)} Ticker, Intervalle {', '.join(config['intervals'])} | Ziel: {data_dir / 'prices'}")
    log_rows: list[dict] = []
    counts = {i: {"ok": 0, "skipped": 0, "failed": []} for i in config["intervals"]}
    aborted = False

    for n, ticker in enumerate(tickers, 1):
        for interval in config["intervals"]:
            out = data_dir / "prices" / interval / f"{ticker}.parquet"
            prefix = f"[{n}/{len(tickers)}] {ticker} {interval}:"
            if is_fresh(out, config["skip_if_younger_than_hours"]):
                counts[interval]["skipped"] += 1
                continue
            try:
                df = save(validate_and_clean(ticker, interval, fetch(ticker, interval)), out, interval)
            except RateLimited:
                print(f"{prefix} ABBRUCH: Yahoo blockiert weitere Anfragen.")
                aborted = True
                break
            except Exception as e:
                msg = str(e) if isinstance(e, DataError) else f"unerwarteter Fehler ({type(e).__name__}): {e}"
                print(f"{prefix} FEHLER: {msg}")
                counts[interval]["failed"].append(ticker)
                log_rows.append({"run": run, "ticker": ticker, "interval": interval, "status": "fehler", "rows": 0, "message": msg})
            else:
                time_column = INTERVALS[interval]["time_column"]
                print(f"{prefix} OK, {len(df)} Zeilen, {df[time_column].iloc[0]} bis {df[time_column].iloc[-1]}")
                counts[interval]["ok"] += 1
                log_rows.append({"run": run, "ticker": ticker, "interval": interval, "status": "ok", "rows": len(df), "message": ""})
            time.sleep(config["pause_seconds"])
        if aborted:
            break

    append_log(data_dir / "import_log.csv", log_rows)

    print("\nZusammenfassung:")
    for interval, c in counts.items():
        failed = c["failed"]
        shown = ", ".join(failed[:20]) + (f" … (+{len(failed) - 20})" if len(failed) > 20 else "")
        print(f"  {interval}: {c['ok']} geladen, {c['skipped']} übersprungen (noch aktuell), {len(failed)} fehlgeschlagen"
              + (f": {shown}" if failed else ""))
    print(f"  Details: {data_dir / 'import_log.csv'}")
    if aborted:
        print("  Import abgebrochen, weil Yahoo drosselt. Später einfach erneut starten – fertige Ticker werden übersprungen.")
    if aborted or any(c["failed"] for c in counts.values()):
        print("  Für fehlgeschlagene Ticker wurde keine vorhandene Datei verändert.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
