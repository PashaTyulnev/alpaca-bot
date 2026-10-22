"""Berechnet Korrelationen der Renditen zwischen den liquidesten Aktien für mehrere Zeitfenster
und erzeugt data/correlations.html (Heatmap, nach Ähnlichkeit sortiert, plus Paarlisten).

Start:  python correlations.py              (500 liquideste Aktien)
        python correlations.py --top 1000   (mehr Aktien, größere Seite)
        python correlations.py --no-open
"""

import argparse
import base64
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from download_data import PROJECT_DIR, load_config, resolve_data_dir

MARKET = "SPY"
# (Kürzel, Beschriftung, Datenquelle, Anzahl Handelstage). Kurze Fenster nutzen Stundenkerzen,
# weil 5 Tagesrenditen für eine Korrelation viel zu wenige Datenpunkte wären.
WINDOWS = [
    ("1W", "1 Woche", "1h", 5),
    ("2W", "2 Wochen", "1h", 10),
    ("1M", "1 Monat", "1h", 21),
    ("3M", "3 Monate", "1d", 63),
    ("1J", "1 Jahr", "1d", 252),
]
MIN_COVERAGE = 0.9  # Aktie braucht Werte für mindestens 90 % der Zeitpunkte im Fenster
MIN_PRICE = 1.0  # Pennystocks ausschließen
TOP_PAIRS = 40


def most_liquid(data_dir: Path, top: int) -> list[str]:
    """Die `top` liquidesten Aktien (ohne SPY): Median des Dollar-Umsatzes der letzten 20 Handelstage."""
    d1 = str(data_dir / "prices" / "1d" / "*.parquet")
    return duckdb.connect().execute(
        """
        WITH recent AS (
            SELECT ticker, close, volume,
                   row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM read_parquet(?, union_by_name=true)
            WHERE date >= current_date - 45
        )
        SELECT ticker FROM recent WHERE rn <= 20 AND ticker <> ?
        GROUP BY ticker
        HAVING median(close) >= ? AND count(*) >= 15
        ORDER BY median(close * volume) DESC
        LIMIT ?
        """,
        [d1, MARKET, MIN_PRICE, top],
    ).df()["ticker"].tolist()


def load_prices(data_dir: Path, top: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    d1 = str(data_dir / "prices" / "1d" / "*.parquet")
    h1 = str(data_dir / "prices" / "1h" / "*.parquet")
    liquid = most_liquid(data_dir, top)
    tickers = liquid + [MARKET]

    daily = con.execute(
        "SELECT ticker, date, adj_close FROM read_parquet(?, union_by_name=true) "
        "WHERE date >= current_date - 400 AND list_contains(?, ticker)",
        [d1, tickers],
    ).df().pivot(index="date", columns="ticker", values="adj_close")
    hourly = con.execute(
        "SELECT ticker, timestamp, close FROM read_parquet(?, union_by_name=true) "
        "WHERE timestamp >= now() - INTERVAL 45 DAY AND list_contains(?, ticker)",
        [h1, tickers],
    ).df().pivot(index="timestamp", columns="ticker", values="close")
    return daily.sort_index(), hourly.sort_index(), liquid


def window_returns(prices: pd.DataFrame, source: str, days: int) -> pd.DataFrame:
    """Log-Renditen der letzten `days` Handelstage. Die erste Stunde eines Tages enthält
    die Kurslücke über Nacht."""
    returns = np.log(prices).diff()
    if source == "1d":
        return returns.iloc[-days:]
    trading_days = pd.Index(returns.index.date).unique()
    keep = pd.Index(returns.index.date).isin(trading_days[-days:])
    return returns[keep]


def market_residuals(returns: pd.DataFrame) -> pd.DataFrame:
    """Entfernt den Marktanteil: Rendite minus Beta × SPY-Rendite (Beta je Aktie im Fenster)."""
    m = returns[MARKET]
    out = {}
    for col in returns.columns.drop(MARKET):
        r = returns[col]
        valid = r.notna() & m.notna()
        mc = m[valid] - m[valid].mean()
        beta = ((r[valid] - r[valid].mean()) * mc).sum() / (mc**2).sum()
        out[col] = r - beta * m
    return pd.DataFrame(out)


def cluster_order(corr: np.ndarray) -> np.ndarray:
    """Hierarchisches Clustering, damit ähnliche Aktien nebeneinander liegen."""
    c = np.nan_to_num(corr, nan=0.0)
    dist = np.sqrt(np.clip(2 * (1 - c), 0, None))
    np.fill_diagonal(dist, 0)
    return leaves_list(linkage(squareform(dist, checks=False), method="average"))


def top_pairs(corr: np.ndarray, tickers: list[str]) -> dict:
    iu = np.triu_indices(len(tickers), k=1)
    vals = corr[iu]
    ok = ~np.isnan(vals)
    vals, a, b = vals[ok], iu[0][ok], iu[1][ok]
    hi, lo = np.argsort(-vals)[:TOP_PAIRS], np.argsort(vals)[:TOP_PAIRS]
    pairs = lambda idx: [[tickers[a[k]], tickers[b[k]], round(float(vals[k]), 3)] for k in idx]
    return {
        "hi": pairs(hi),
        "lo": pairs(lo),
        "mean": round(float(vals.mean()), 3),
        "share_hi": round(float((vals > 0.7).mean()), 4),
        "share_lo": round(float((vals < -0.3).mean()), 4),
    }


def matrix_payload(returns: pd.DataFrame) -> dict:
    corr_df = returns.corr(min_periods=int(len(returns) * MIN_COVERAGE))
    corr = corr_df.to_numpy()
    order = cluster_order(corr)
    tickers = [corr_df.columns[i] for i in order]
    corr = corr[np.ix_(order, order)]
    # Kompakt speichern: Korrelation × 100 als int8, fehlend = -128
    q = np.where(np.isnan(corr), -128, np.round(corr * 100)).astype(np.int8)
    return {"tickers": tickers, "m": base64.b64encode(q.tobytes()).decode("ascii"), **top_pairs(corr, tickers)}


def compute(daily: pd.DataFrame, hourly: pd.DataFrame) -> list[dict]:
    results = []
    for key, label, source, days in WINDOWS:
        prices = hourly if source == "1h" else daily
        rets = window_returns(prices, source, days)
        # Nur Aktien mit ausreichend Daten und echter Kursbewegung (z. B. keine SPAC-Hüllen bei 10 $)
        coverage = rets.notna().mean()
        moving = rets.std() > 1e-6
        rets = rets.loc[:, (coverage >= MIN_COVERAGE) & moving]
        if MARKET not in rets.columns:
            raise SystemExit(f"{MARKET} fehlt im Fenster {label} – wird für die Marktbereinigung benötigt.")
        start, end = rets.index[0], rets.index[-1]
        fmt = "%d.%m.%Y %H:%M" if source == "1h" else "%d.%m.%Y"
        results.append({
            "key": key,
            "label": label,
            "source": source,
            "n_obs": len(rets),
            "range": f"{pd.Timestamp(start):{fmt}} – {pd.Timestamp(end):{fmt}}",
            "raw": matrix_payload(rets.drop(columns=MARKET)),
            "adj": matrix_payload(market_residuals(rets)),
        })
        print(f"{label}: {rets.shape[1] - 1} Aktien, {len(rets)} {'Stunden' if source == '1h' else 'Tages'}-Renditen")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Korrelations-Heatmap der liquidesten Aktien erzeugen.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml", help="Pfad zur TOML-Konfiguration")
    parser.add_argument("--top", type=int, default=500, help="Anzahl der liquidesten Aktien (Standard 500)")
    parser.add_argument("--no-open", action="store_true", help="Seite nur erzeugen, nicht im Browser öffnen")
    args = parser.parse_args()

    data_dir = resolve_data_dir(load_config(args.config))
    daily, hourly, liquid = load_prices(data_dir, args.top)
    print(f"{len(liquid)} liquideste Aktien ausgewählt (Kurs ≥ {MIN_PRICE:.0f} $, nach Dollar-Umsatz)")

    names = {}
    universe_file = data_dir / "universe" / "nasdaq-common.csv"
    if universe_file.exists():
        names = dict(pd.read_csv(universe_file, dtype=str, keep_default_na=False).values)

    payload = {
        "windows": compute(daily, hourly),
        "names": {t: names.get(t, "") for t in liquid},
        "created": f"{datetime.now():%d.%m.%Y %H:%M}",
        "top": len(liquid),
    }
    template = (PROJECT_DIR / "correlations_template.html").read_text(encoding="utf-8")
    # "</" maskieren, damit Namen im JSON den <script>-Block nicht beenden können
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = data_dir / "correlations.html"
    out.write_text(template.replace("/*__DATA__*/null", data_json), encoding="utf-8")
    print(f"Seite geschrieben: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
