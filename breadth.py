"""Marktbreite als Signal für SPY (Intraday, Alpaca-5-Minuten-Kerzen) -> data/breadth.html

VORAB FESTGELEGTE REGEL (vor der ersten Berechnung notiert, danach nicht verändert):
  * Eröffnungsspanne je Aktie = Hoch/Tief der ersten 30 Minuten.
  * Ab 10:00 wird nach jeder 5-Min-Kerze gezählt, welcher Anteil der Aktien unter dem Tief (bzw. über dem
    Hoch) seiner eigenen Eröffnungsspanne schließt.
  * Signal "Breite unten": erstmals am Tag mindestens 50 % der Aktien unter ihrer Spanne, spätestens bis 15:00.
    -> SPY kaufen zur Eröffnung der nächsten Kerze, verkaufen zum Tagesschluss.
  * Signal "Breite oben": erstmals mindestens 50 % über ihrer Spanne -> spiegelbildlich SPY leerverkaufen.
  * Kosten: 2 bps je Hin- und Rückweg.
  * Vergleich: SPY von derselben Uhrzeit bis Schluss an allen Tagen (Grundrauschen der Tageszeit).

WICHTIG: Diese Idee entstand beim Betrachten derselben Daten (2016–2026). Es gibt deshalb keinen sauberen
Prüfzeitraum. Ein echter Test ist nur mit NEUEN Daten möglich (Paper-Trading ab jetzt).

Start:  python breadth.py
"""

import argparse
import html
import webbrowser
from datetime import datetime, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from correlations import MARKET
from download_data import PROJECT_DIR, load_config, resolve_data_dir

THRESHOLD = 0.50
LAST_SIGNAL = time(15, 0)
COST_BPS = 2.0
OOS_START = pd.Timestamp("2023-01-01")
MIN_COVERAGE = 0.8  # Zeitpunkt nur werten, wenn mind. 80 % der Aktien des Tages eine Kerze haben

BREADTH_QUERY = """
WITH b AS (
    SELECT ticker, timestamp AS ts, CAST(timestamp AS DATE) AS d, high, low, close
    FROM read_parquet(?, union_by_name=true) WHERE session = 'regular' AND ticker <> ?
),
t0 AS (SELECT d, min(ts) AS t_open FROM b GROUP BY d),
orr AS (
    SELECT ticker, b.d, max(high) AS orh, min(low) AS orl
    FROM b JOIN t0 USING (d) WHERE ts < t_open + INTERVAL 30 MINUTE GROUP BY ticker, b.d
)
SELECT b.ts, b.d, count(*) AS n, sum(CAST(close < orl AS INT)) AS n_below, sum(CAST(close > orh AS INT)) AS n_above,
       any_value(t0.t_open) AS t_open
FROM b JOIN orr USING (ticker, d) JOIN t0 USING (d)
WHERE ts >= t_open + INTERVAL 30 MINUTE
GROUP BY b.ts, b.d ORDER BY b.ts
"""


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    files = [str(f) for f in (data_dir / "alpaca" / "5Min").glob("*.parquet")]
    breadth = con.execute(BREADTH_QUERY, [files, MARKET]).df()
    spy = con.execute(
        "SELECT timestamp AS ts, CAST(timestamp AS DATE) AS d, open, close FROM read_parquet(?) WHERE session = 'regular' ORDER BY ts",
        [str(data_dir / "alpaca" / "5Min" / f"{MARKET}.parquet")],
    ).df()
    return breadth, spy


def signals(breadth: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    b = breadth.copy()
    # Zeitstempel = Kerzenbeginn. Die Kerze 9:55 gehört noch zur Eröffnungsspanne; erste Ausbruchskerze ist 10:00
    b = b[b["ts"] >= b["t_open"] + pd.Timedelta(minutes=30)]
    max_n = b.groupby("d")["n"].transform("max")
    b = b[(b["n"] >= MIN_COVERAGE * max_n) & (b["ts"].dt.time <= LAST_SIGNAL)]
    b["below"] = b["n_below"] / b["n"]
    b["above"] = b["n_above"] / b["n"]

    spy_open = spy.set_index("ts")["open"]
    spy_close = spy.groupby("d")["close"].last()
    rows = []
    for side, col, sign in (("unten", "below", 1), ("oben", "above", -1)):
        hit = b[b[col] >= THRESHOLD].groupby("d").first()
        # Signal bei Schluss der Kerze (ts + 5 Min) -> Einstieg zur Eröffnung der nächsten Kerze, die dann beginnt
        entry_ts = hit["ts"] + pd.Timedelta(minutes=5)
        entry = spy_open.reindex(entry_ts).to_numpy()
        ret = spy_close.reindex(hit.index).to_numpy() / entry - 1
        rows.append(pd.DataFrame({"d": hit.index, "side": side, "signal_time": (hit["ts"] + pd.Timedelta(minutes=5)).dt.time,
                                  "breadth": hit[col].to_numpy(), "spy_ret": ret, "trade_ret": sign * ret - COST_BPS / 1e4}))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["spy_ret"])
    out["period"] = np.where(out["d"] >= OOS_START, "ab 2023", "2016–2022")
    out["year"] = pd.to_datetime(out["d"]).dt.year
    return out


def baseline(spy: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """SPY ab derselben Uhrzeit bis Schluss an ALLEN Tagen: zeigt, ob der Effekt nur die Tageszeit ist."""
    s = spy.copy()
    s["t"] = s["ts"].dt.time
    close = s.groupby("d")["close"].last()
    out = {}
    for side in ("unten", "oben"):
        times = sig.loc[sig["side"] == side, "signal_time"]
        typical = times.mode().iloc[0] if len(times) else time(11, 0)  # Kerze, die zur Signalzeit beginnt
        entry = s[s["t"] == typical].set_index("d")["open"]
        r = close.reindex(entry.index) / entry - 1
        out[side] = {"time": typical, "mean": r.mean() * 1e4, "n": r.notna().sum()}
    return out


def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 10:
        return {"n": len(x), "mean": np.nan, "t": np.nan, "hit": np.nan, "sum": np.nan}
    return {"n": len(x), "mean": x.mean() * 1e4, "t": x.mean() / (x.std() / np.sqrt(len(x))), "hit": (x > 0).mean(),
            "sum": ((1 + x).prod() - 1)}


def f(v, d=1, unit="") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.{d}f}" + unit).replace(".", ",").replace("-", "−")


def render(sig: pd.DataFrame, base: dict, n_tickers: int) -> str:
    rows, yrows = [], []
    years = sorted(sig["year"].unique())
    for side, label in (("unten", "Breite unten → SPY kaufen"), ("oben", "Breite oben → SPY leerverkaufen")):
        s = sig[sig["side"] == side]
        for per in ("2016–2022", "ab 2023"):
            st = summarize(s.loc[s["period"] == per, "trade_ret"])
            rows.append(f"<tr><td>{label if per == '2016–2022' else ''}</td><td class='muted'>{per}</td><td class='n'>{st['n']}</td>"
                        f"<td class='n'><b>{f(st['mean'])}</b></td><td class='n'>{f(st['t'])}</td>"
                        f"<td class='n'>{'–' if np.isnan(st['hit']) else f'{st['hit']:.0%}'}</td><td class='n'>{f(st['sum'] * 100 if not np.isnan(st['sum']) else np.nan, 0, ' %')}</td></tr>")
        y = s.groupby("year")["trade_ret"].agg(["mean", "count"])
        cells = "".join(
            f"<td class='n' title='{int(y.loc[yr, 'count'])} Tage' style='background:rgba({'var(--pos-rgb)' if y.loc[yr, 'mean'] > 0 else 'var(--neg-rgb)'},{min(abs(y.loc[yr, 'mean']) * 1e4 / 40, 1) * 0.5:.2f})'>{f(y.loc[yr, 'mean'] * 1e4)}</td>"
            if yr in y.index else "<td class='n muted'>–</td>" for yr in years)
        yrows.append(f"<tr><td>{label}</td>{cells}</tr>")
    b = base
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marktbreite</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --accent:#2f6fde; --pos-rgb:31,138,76; --neg-rgb:227,73,72; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30; --accent:#5b8ff0; --pos-rgb:76,194,126; --neg-rgb:230,103,103; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1000px; margin:0 auto; }} a {{ color:var(--accent); text-decoration:none; }}
h1 {{ font-size:22px; margin:4px 0; }} h2 {{ font-size:17px; margin:0 0 8px; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }}
.warn {{ border-left:3px solid #e0a040; }}
.scroll {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ font-size:11px; font-weight:600; color:var(--muted); }} .n {{ text-align:right; font-variant-numeric:tabular-nums; }}
li {{ margin:4px 0; }}
</style></head><body><main>
<a href="overview.html">← Datenübersicht</a> · <a href="analysis.html">Eröffnungs-Analysen</a> · <a href="backtests.html">Tagesstrategien</a>
<h1>Marktbreite als Signal für SPY</h1>
<p class="muted">{n_tickers} liquide Aktien · 5-Minuten-Kerzen (Alpaca) · Kosten {COST_BPS:.0f} bps je Trade</p>
<div class="card"><h2>Regel (vorab festgelegt)</h2><ul class="muted">
<li>Eröffnungsspanne je Aktie = Hoch und Tief der ersten 30 Minuten.</li>
<li>Ab 10:00 wird nach jeder 5-Min-Kerze gezählt, welcher Anteil der Aktien unter (bzw. über) seiner Spanne schließt.</li>
<li>Erstmals ≥ {THRESHOLD:.0%} darunter (bis 15:00) → SPY zur nächsten Kerze kaufen, zum Schluss verkaufen. Spiegelbildlich ≥ {THRESHOLD:.0%} darüber → SPY leerverkaufen.</li>
</ul></div>
<div class="card warn"><b>Kein sauberer Test möglich.</b> <span class="muted">Die Idee entstand beim Betrachten genau dieser Daten. Auch der Zeitraum ab 2023 ist deshalb nicht unabhängig. Ein echter Test geht nur mit Daten, die es heute noch nicht gibt: Paper-Trading ab jetzt.</span></div>
<div class="card"><h2>Ergebnis je Trade (nach Kosten)</h2><div class="scroll"><table>
<tr><th>Regel</th><th>Zeitraum</th><th class='n'>Tage mit Signal</th><th class='n'>Ø bps</th><th class='n'>t-Wert</th><th class='n'>Treffer</th><th class='n'>Summe</th></tr>
{''.join(rows)}</table></div>
<p class="muted">Vergleich Tageszeit-Effekt: SPY von {b['unten']['time']:%H:%M} bis Schluss an <i>allen</i> Tagen: {f(b['unten']['mean'])} bps im Schnitt.
Erst was deutlich darüber liegt, kommt vom Signal und nicht von der Uhrzeit.</p></div>
<div class="card"><h2>Je Jahr (Ø bps nach Kosten)</h2><div class="scroll"><table>
<tr><th>Regel</th>{''.join(f"<th class='n'>{y}</th>" for y in years)}</tr>{''.join(yrows)}</table></div></div>
<p class="muted" style="font-size:12px">Erzeugt am {datetime.now():%d.%m.%Y %H:%M} mit breadth.py</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Marktbreite als Signal für SPY.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    data_dir = resolve_data_dir(load_config(args.config))
    breadth, spy = load(data_dir)
    sig = signals(breadth, spy)
    base = baseline(spy, sig)
    n_tickers = len(list((data_dir / "alpaca" / "5Min").glob("*.parquet"))) - 1
    for side in ("unten", "oben"):
        s = sig[sig["side"] == side]
        for per in ("2016–2022", "ab 2023"):
            st = summarize(s.loc[s["period"] == per, "trade_ret"])
            print(f"Breite {side:<5} {per}: {st['n']:4d} Tage, Ø {st['mean']:+6.1f} bps, t {st['t']:+5.1f}, Treffer {st['hit']:.0%}")
        print(f"  Signalzeit Median: {pd.Series([datetime.combine(datetime(2000,1,1), t) for t in s['signal_time']]).median():%H:%M}"
              f" | Vergleich SPY ab {base[side]['time']:%H:%M} an allen Tagen: {base[side]['mean']:+.1f} bps")
    out = data_dir / "breadth.html"
    out.write_text(render(sig, base, n_tickers), encoding="utf-8")
    print(f"Seite geschrieben: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
