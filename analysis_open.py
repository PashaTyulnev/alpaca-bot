"""Ereignis-Analysen rund um die Börseneröffnung auf Basis der Alpaca-5-Minuten-Kerzen.

Fragen:
  1. Eröffnungslücke (Gap): Läuft der Kurs nach einer Lücke im Tagesverlauf weiter oder schließt er sie?
  2. Lage um 10:00 relativ zum Vortageshoch/-tief: Was passiert im Rest des Tages?
  3. Opening Range Breakout (30 Min): Ausbruch über das Hoch / unter das Tief der ersten 30 Minuten.

Ehrlicher Test: Effekt wird 2016–2022 gemessen (In-Sample). Die Handelsrichtung ergibt sich aus diesem
Zeitraum und wird ab 2023 (Out-of-Sample) unverändert nach Kosten geprüft.

Start:  python analysis_open.py              (schreibt data/analysis.html)
        python analysis_open.py --cost-bps 10
"""

import argparse
import html
import webbrowser
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from correlations import MARKET
from download_data import PROJECT_DIR, load_config, resolve_data_dir

OOS_START = date(2023, 1, 1)
OPENING_RANGE_MIN = 30
MIN_BARS_PER_DAY = 60  # volle Tage haben 78 Kerzen à 5 Min; halbe Handelstage und Datenlücken fallen raus
MAX_ABS_GAP = 0.25
WINSOR_Q = 0.005  # Renditen auf das 0,5- bzw. 99,5-%-Quantil kappen
HEDGE_COST_BPS = 1.0  # Zusatzkosten für die SPY-Absicherung
MIN_NET_BPS = 2.0  # Mindestgewinn nach Kosten, damit ein Effekt nicht nur knapp an der Kostengrenze liegt

DAY_QUERY = f"""
WITH b AS (
    SELECT timestamp AS ts, CAST(timestamp AS DATE) AS d, open, high, low, close,
           -- Einstieg erst zur Eröffnung der Kerze NACH dem Signal (das Signal kennt man erst bei Kerzenschluss)
           lead(open) OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp) AS next_open,
           lead(timestamp) OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp) AS next_ts
    FROM read_parquet(?) WHERE session = 'regular'
),
day AS (
    SELECT d, min(ts) AS t_open, arg_min(open, ts) AS o, arg_max(close, ts) AS c,
           max(high) AS h, min(low) AS l, count(*) AS n
    FROM b GROUP BY d
),
orr AS (
    SELECT b.d, max(b.high) AS or_h, min(b.low) AS or_l, arg_max(b.close, b.ts) AS p_or
    FROM b JOIN day USING (d)
    WHERE b.ts < day.t_open + INTERVAL {OPENING_RANGE_MIN} MINUTE
    GROUP BY b.d
),
brk AS (
    SELECT b.d,
           arg_min(b.open, b.ts) AS entry10, min(b.ts) AS t10,
           arg_min(b.next_ts, b.ts) FILTER (WHERE b.close > orr.or_h) AS long_t,
           arg_min(b.next_ts, b.ts) FILTER (WHERE b.close < orr.or_l) AS short_t,
           arg_min(b.next_open, b.ts) FILTER (WHERE b.close > orr.or_h) AS long_entry,
           arg_min(b.next_open, b.ts) FILTER (WHERE b.close < orr.or_l) AS short_entry
    FROM b JOIN day USING (d) JOIN orr USING (d)
    WHERE b.ts >= day.t_open + INTERVAL {OPENING_RANGE_MIN} MINUTE
    GROUP BY b.d
)
SELECT day.d, day.o, day.c, day.h, day.l, day.n, orr.or_h, orr.or_l, orr.p_or, brk.entry10, brk.t10, brk.long_entry, brk.long_t, brk.short_entry, brk.short_t,
       lag(day.d) OVER w AS prev_d, lag(day.c) OVER w AS prev_c, lag(day.h) OVER w AS prev_h, lag(day.l) OVER w AS prev_l
FROM day JOIN orr USING (d) LEFT JOIN brk USING (d)
WINDOW w AS (ORDER BY day.d)
ORDER BY day.d
"""


def build_day_table(data_dir: Path) -> pd.DataFrame:
    """Eine Zeile je Aktie und Handelstag. Wird zusätzlich als Parquet gespeichert (für spätere Backtests)."""
    files = sorted((data_dir / "alpaca" / "5Min").glob("*.parquet"))
    if not files:
        raise SystemExit("Keine Alpaca-Daten gefunden. Erst alpaca_download.py ausführen.")
    cal = pd.read_csv(data_dir / "alpaca" / "calendar.csv", parse_dates=["date"])
    prev_trading_day = dict(zip(cal["date"].dt.date.iloc[1:], cal["date"].dt.date.iloc[:-1]))

    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    parts = []
    for f in files:
        df = con.execute(DAY_QUERY, [str(f)]).df()
        df.insert(0, "ticker", f.stem)
        parts.append(df)
    days = pd.concat(parts, ignore_index=True)
    days["d"] = pd.to_datetime(days["d"]).dt.date
    days["prev_d"] = pd.to_datetime(days["prev_d"]).dt.date
    # Vortageswerte nur verwenden, wenn wirklich der vorherige Handelstag vorliegt
    days = days[(days["n"] >= MIN_BARS_PER_DAY) & (days["prev_d"] == days["d"].map(prev_trading_day))]
    days.to_parquet(data_dir / "alpaca" / "day_stats_5Min.parquet", index=False)
    return days


def spy_opens(data_dir: Path) -> pd.Series:
    """Eröffnungskurs jeder regulären SPY-5-Min-Kerze, um den Markt ab exakt demselben Zeitpunkt zu messen."""
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    path = str(data_dir / "alpaca" / "5Min" / f"{MARKET}.parquet")
    return con.execute("SELECT timestamp, open FROM read_parquet(?) WHERE session = 'regular'", [path]).df().set_index("timestamp")["open"]


def add_features(days: pd.DataFrame, spy_open: pd.Series) -> pd.DataFrame:
    df = days.copy()
    df["gap"] = df["o"] / df["prev_c"] - 1
    df["r_day"] = df["c"] / df["o"] - 1  # Eröffnung -> Schluss
    df["r_rest"] = df["c"] / df["entry10"] - 1  # Eröffnung der 10:00-Kerze -> Schluss
    df["r_long"] = df["c"] / df["long_entry"] - 1
    df["r_short"] = df["c"] / df["short_entry"] - 1  # Kursbewegung nach Ausbruch nach unten

    spy = df[df["ticker"] == MARKET].set_index("d")
    df["spy_day"] = df["d"].map(spy["r_day"])
    df["spy_rest"] = df["d"].map(spy["r_rest"])
    # SPY vom Einstiegszeitpunkt des Ausbruchs bis Schluss (entspricht einer Absicherung mit SPY)
    spy_close = df["d"].map(spy["c"])
    df["spy_long"] = spy_close / spy_open.reindex(df["long_t"]).to_numpy() - 1
    df["spy_short"] = spy_close / spy_open.reindex(df["short_t"]).to_numpy() - 1
    # Lücken über ±25 % sind meist Kapitalmaßnahmen (z. B. Abspaltungen) oder Ausnahmeereignisse
    df = df[(df["ticker"] != MARKET) & (df["gap"].abs() <= MAX_ABS_GAP)].copy()
    df["period"] = np.where(df["d"] >= OOS_START, "oos", "is")
    df["year"] = pd.to_datetime(df["d"]).dt.year
    return df


def winsorize(v: pd.Series) -> pd.Series:
    """Kappt extreme Renditen, damit einzelne Ausreißer (News, Datenfehler) den Schnitt nicht bestimmen."""
    lo, hi = v.quantile([WINSOR_Q, 1 - WINSOR_Q])
    return v.clip(lo, hi)


def stats(values: pd.Series, days: pd.Series) -> dict:
    """Kennzahlen je Gruppe. Mittelwert und t-Wert beruhen auf Tagesmitteln: Jeder Tag zählt gleich,
    wie bei einem Portfolio, das jeden Tag alle Signale gleich gewichtet handelt. Ereignisse am selben
    Tag hängen stark zusammen; einzeln gezählt wirkte ein Effekt viel sicherer, als er ist."""
    v = winsorize(values.dropna())
    if len(v) < 30:
        return {"n": len(v), "days": 0, "mean": np.nan, "median": np.nan, "hit": np.nan, "t": np.nan}
    daily = v.groupby(days[v.index]).mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    return {"n": len(v), "days": len(daily), "mean": daily.mean() * 1e4, "median": v.median() * 1e4,
            "hit": (v > 0).mean(), "t": t}


def yearly_stats(sub: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    """Ø je Jahr nach derselben Methode (Tagesmittel, gekappt)."""
    v = winsorize(sub[ret_col].dropna())
    daily = v.groupby(sub.loc[v.index, "d"]).mean()
    years = pd.to_datetime(pd.Series(daily.index)).dt.year.to_numpy()
    out = pd.DataFrame({"mean": daily.groupby(years).mean() * 1e4,
                        "count": v.groupby(sub.loc[v.index, "year"]).size()})
    return out


def analyse(df: pd.DataFrame, group_col: str, ret_col: str, adj_col: str, groups: list[str], cost: float) -> dict:
    """Bewertet je Gruppe die MARKTBEREINIGTE Rendite (Aktie minus SPY im selben Zeitfenster). Das entspricht
    einem Trade, der mit SPY abgesichert ist; so zählt nur, was die Aktie selbst leistet. Die rohe Rendite
    wird zum Vergleich mit ausgegeben."""
    rows, yearly = [], {}
    total_cost = cost + HEDGE_COST_BPS
    for g in groups:
        sub = df[df[group_col] == g].copy()
        sub["_x"] = sub[ret_col] - sub[adj_col]
        is_m, oos_m = sub["period"] == "is", sub["period"] == "oos"
        ins, oos = stats(sub.loc[is_m, "_x"], sub["d"]), stats(sub.loc[oos_m, "_x"], sub["d"])
        raw = {"is": stats(sub.loc[is_m, ret_col], sub["d"])["mean"], "oos": stats(sub.loc[oos_m, ret_col], sub["d"])["mean"]}
        # Handelsrichtung aus In-Sample, dann Out-of-Sample unverändert nach Kosten prüfen
        direction = np.sign(ins["mean"]) if not np.isnan(ins["mean"]) else 0
        oos_net = direction * oos["mean"] - total_cost if direction else np.nan
        robust = bool(direction != 0 and not np.isnan(oos["t"]) and abs(ins["t"]) > 2 and np.sign(oos["mean"]) == direction
                      and abs(oos["t"]) > 2 and oos_net > MIN_NET_BPS)
        rows.append({"group": g, "is": ins, "oos": oos, "raw": raw, "direction": direction, "oos_net": oos_net, "robust": robust})
        yearly[g] = yearly_stats(sub, "_x")
    return {"rows": rows, "yearly": yearly}


# ---------- HTML ----------

def f_bps(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1f}".replace(".", ",").replace("-", "−")


def f_int(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def column_chart(result: dict, cost: float) -> str:
    """Gruppierte Säulen: In-Sample (blau) und Out-of-Sample (orange), Ø Rendite je Ereignis in bps."""
    rows = result["rows"]
    vals = [r[p]["mean"] for r in rows for p in ("is", "oos") if not np.isnan(r[p]["mean"])]
    peak = max([abs(v) for v in vals] + [cost * 2, 5])
    # Glatte Achsenschritte (1, 2, 5 × 10^k), etwa zwei Linien je Seite
    step = min(s * 10 ** k for k in range(-1, 5) for s in (1, 2, 5) if s * 10 ** k >= peak / 2)
    lim = step * 2 * 1.08
    w, h, top, bottom, left = 720, 240, 16, 44, 44
    plot_h = h - top - bottom
    y = lambda v: top + plot_h / 2 - v / lim * plot_h / 2
    band = (w - left - 8) / len(rows)
    bar = min(24, band / 3)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Ø Rendite je Gruppe">']
    # Kostenband: Effekte innerhalb dieses Bereichs sind nach Kosten nicht handelbar
    c = cost + HEDGE_COST_BPS
    parts.append(f'<rect x="{left}" y="{y(c):.1f}" width="{w - left - 8}" height="{y(-c) - y(c):.1f}" class="costband"/>')
    for v in (2 * step, step, 0, -step, -2 * step):
        parts.append(f'<line x1="{left}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if v == 0 else "grid"}"/>')
        parts.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="axis" text-anchor="end">{f"{v:+g}".replace("-", "−") if v else "0"}</text>')
    for i, r in enumerate(rows):
        cx = left + band * i + band / 2
        for k, (p, cls) in enumerate((("is", "s1"), ("oos", "s2"))):
            v = r[p]["mean"]
            if np.isnan(v):
                continue
            x = cx - bar - 1 + k * (bar + 2)
            y0, y1 = y(0), y(v)
            hgt = max(abs(y1 - y0), 1)
            ry = min(4, hgt / 2)
            # Abgerundetes Datenende, flache Kante an der Nulllinie
            if v >= 0:
                d = f"M{x:.1f},{y0:.1f} V{y1 + ry:.1f} Q{x:.1f},{y1:.1f} {x + ry:.1f},{y1:.1f} H{x + bar - ry:.1f} Q{x + bar:.1f},{y1:.1f} {x + bar:.1f},{y1 + ry:.1f} V{y0:.1f} Z"
            else:
                d = f"M{x:.1f},{y0:.1f} V{y1 - ry:.1f} Q{x:.1f},{y1:.1f} {x + ry:.1f},{y1:.1f} H{x + bar - ry:.1f} Q{x + bar:.1f},{y1:.1f} {x + bar:.1f},{y1 - ry:.1f} V{y0:.1f} Z"
            label = "2016–2022" if p == "is" else "ab 2023"
            parts.append(f'<path d="{d}" class="{cls}"><title>{html.escape(r["group"])} · {label}: {f_bps(v)} bps · {f_int(r[p]["n"])} Ereignisse</title></path>')
        parts.append(f'<text x="{cx:.1f}" y="{h - bottom + 18}" class="axis" text-anchor="middle">{html.escape(r["group"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def result_table(result: dict, cost: float) -> str:
    head = ("<tr><th>Gruppe</th><th class='n'>Ereignisse<br>16–22 / ab 23</th><th class='n'>Ø bps<br>16–22</th>"
            "<th class='n'>t-Wert<br>16–22</th><th class='n'>Ø bps<br>ab 23</th><th class='n'>t-Wert<br>ab 23</th>"
            "<th class='n'>Treffer<br>ab 23</th><th class='n'>roh, ohne<br>Bereinigung ab 23</th><th class='n'>Test ab 23<br>netto bps</th><th>Urteil</th></tr>")
    body = []
    for r in result["rows"]:
        i, o = r["is"], r["oos"]
        verdict = ('<span class="tag ok">robust</span>' if r["robust"] else
                   '<span class="tag">nicht belastbar</span>')
        side = {1: "long", -1: "short"}.get(int(r["direction"]), "–") if r["direction"] == r["direction"] else "–"
        body.append(
            f"<tr><td>{html.escape(r['group'])}</td><td class='n'>{f_int(i['n'])} / {f_int(o['n'])}</td>"
            f"<td class='n'>{f_bps(i['mean'])}</td><td class='n'>{f_bps(i['t'])}</td>"
            f"<td class='n'>{f_bps(o['mean'])}</td><td class='n'>{f_bps(o['t'])}</td>"
            f"<td class='n'>{'–' if np.isnan(o['hit']) else f'{o['hit'] * 100:.0f} %'}</td>"
            f"<td class='n muted'>{f_bps(r['raw']['oos'])}</td>"
            f"<td class='n'>{f_bps(r['oos_net'])} <span class='muted'>({side})</span></td><td>{verdict}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def yearly_table(result: dict) -> str:
    years = sorted({y for t in result["yearly"].values() for y in t.index})
    head = "<tr><th>Gruppe</th>" + "".join(f"<th class='n'>{y}</th>" for y in years) + "</tr>"
    body = []
    for g, t in result["yearly"].items():
        cells = []
        for y in years:
            if y in t.index and t.loc[y, "count"] >= 30:
                v = t.loc[y, "mean"]
                alpha = min(abs(v) / 30, 1) * 0.55
                color = "var(--pos-rgb)" if v > 0 else "var(--neg-rgb)"
                cells.append(f"<td class='n' style='background:rgba({color},{alpha:.2f})' title='{f_int(t.loc[y, 'count'])} Ereignisse'>{f_bps(v)}</td>")
            else:
                cells.append("<td class='n muted'>–</td>")
        body.append(f"<tr><td>{html.escape(g)}</td>{''.join(cells)}</tr>")
    return f"<table class='years'>{head}{''.join(body)}</table>"


def section(title: str, question: str, result: dict, note: str, cost: float) -> str:
    return f"""<section class="card">
<h2>{title}</h2><p class="muted">{question}</p>
<div class="legend"><span><i class="sw s1"></i>2016–2022 (Finden)</span><span><i class="sw s2"></i>ab 2023 (Prüfen)</span>
<span><i class="sw band"></i>Kostenbereich ±{cost + HEDGE_COST_BPS:.0f} bps</span></div>
{column_chart(result, cost)}
<div class="scroll">{result_table(result, cost)}</div>
<details><summary>Stabilität je Jahr (Ø bps)</summary><div class="scroll">{yearly_table(result)}</div></details>
<p class="note">{note}</p>
</section>"""


def render(df: pd.DataFrame, analyses: list[tuple], cost: float) -> str:
    n_tickers = df["ticker"].nunique()
    first, last = min(df["d"]), max(df["d"])
    sections = "\n".join(section(*a, cost) for a in analyses)
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eröffnungs-Analysen</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --grid:#ececea; --accent:#2f6fde;
        --s1:#2a78d6; --s2:#eb6834; --band:rgba(107,107,112,.10); --ok:#1f8a4c; --pos-rgb:227,73,72; --neg-rgb:42,120,214; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30; --grid:#26262a; --accent:#5b8ff0;
          --s1:#3987e5; --s2:#d95926; --band:rgba(154,154,160,.12); --ok:#4cc27e; --pos-rgb:230,103,103; --neg-rgb:57,135,229; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; }}
a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
h1 {{ font-size:22px; margin:4px 0; }} h2 {{ font-size:17px; margin:0 0 4px; }}
.muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }}
.intro li {{ margin:4px 0; }}
.chart {{ width:100%; height:auto; display:block; margin:8px 0 4px; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }} .chart .zero {{ stroke:var(--muted); stroke-width:1; }}
.chart .axis {{ fill:var(--muted); font-size:11px; }}
.chart .s1 {{ fill:var(--s1); }} .chart .s2 {{ fill:var(--s2); }} .chart .costband {{ fill:var(--band); }}
.chart path:hover {{ opacity:.8; }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:var(--muted); margin-top:8px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.sw.s1 {{ background:var(--s1); }} .sw.s2 {{ background:var(--s2); }} .sw.band {{ background:var(--band); border:1px solid var(--line); }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ font-size:11px; font-weight:600; color:var(--muted); vertical-align:bottom; }}
.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
.tag {{ display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px; border:1px solid currentColor; color:var(--muted); }}
.tag.ok {{ color:var(--ok); }}
details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--muted); }}
.note {{ font-size:13px; color:var(--muted); margin:10px 0 0; }}
footer {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style></head>
<body><main>
<a href="overview.html">← Datenübersicht</a> · <a href="correlations.html">Korrelationen</a> · <a href="backtests.html">Tagesstrategien</a> · <a href="breadth.html">Marktbreite</a>
<h1>Eröffnungs-Analysen</h1>
<p class="muted">{f_int(n_tickers)} liquide Aktien · {f_int(len(df))} Handelstage · {first:%d.%m.%Y} – {last:%d.%m.%Y} · 5-Minuten-Kerzen (Alpaca, Gesamtmarkt)</p>
<div class="card intro"><b>So liest du die Ergebnisse</b><ul class="muted">
<li><b>bps</b> = Basispunkte, 1 bps = 0,01 %. Ein Ø von +10 bps heißt: im Schnitt +0,1 % je Ereignis. Der Ø wird je Handelstag gebildet (alle Ereignisse eines Tages gleich gewichtet, dann jeder Tag gleich), extreme Renditen sind auf das 0,5-/99,5-%-Quantil gekappt.</li>
<li>Jede Frage wird in <b>2016–2022</b> gemessen (blau). Daraus folgt die Handelsrichtung (long oder short). Ab <b>2023</b> (orange) wird genau diese Richtung ohne Anpassung geprüft.</li>
<li><b>Alle Werte sind marktbereinigt</b>: Rendite der Aktie minus SPY im selben Zeitfenster. Das entspricht einem mit SPY abgesicherten Trade und zeigt, was die Aktie selbst leistet – nicht der Markt. Die rohe Rendite steht grau daneben.</li>
<li><b>Netto</b> zieht {cost:.0f} bps für Hin- und Rückweg plus {HEDGE_COST_BPS:.0f} bps für die Absicherung ab (Spread und Slippage). Alles im grauen Kostenband ist nach Kosten nicht handelbar.</li>
<li><b>t-Wert</b>: Ab etwa |2| ist ein Effekt eher kein Zufall. Er wird über Tagesmittel berechnet, weil Aktien am selben Tag gemeinsam schwanken.</li>
<li><b>Robust</b> heißt nur: in beiden Zeiträumen gleiche Richtung, beide |t| &gt; 2 und ab 2023 nach Kosten mindestens +{MIN_NET_BPS:.0f} bps. Das ist ein Kandidat für einen echten Backtest, noch keine Strategie.</li>
<li>Survivorship Bias: Untersucht werden die heute liquidesten Aktien. Firmen, die seit 2016 verschwunden sind, fehlen.</li>
</ul></div>
{sections}
<footer>Erzeugt am {datetime.now():%d.%m.%Y %H:%M} mit analysis_open.py · Eröffnungsspanne = erste {OPENING_RANGE_MIN} Minuten · Tage mit weniger als {MIN_BARS_PER_DAY} Kerzen (z. B. halbe Handelstage) ausgeschlossen. Lücken über ±{MAX_ABS_GAP * 100:.0f} % (meist Kapitalmaßnahmen) ausgeschlossen.</footer>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Eröffnungs-Analysen aus Alpaca-5-Minuten-Kerzen.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Kosten je Hin- und Rückweg in bps (Standard 5)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(load_config(args.config))
    days = build_day_table(data_dir)
    if MARKET not in set(days["ticker"]):
        raise SystemExit(f"{MARKET} fehlt in den Alpaca-Daten – wird als Marktvergleich benötigt.")
    df = add_features(days, spy_opens(data_dir))
    print(f"{df['ticker'].nunique()} Aktien, {len(df):,} Aktien-Tage".replace(",", "."))
    cost = args.cost_bps

    gap_labels = ["< −3 %", "−3 bis −1 %", "−1 bis −0,25 %", "±0,25 %", "+0,25 bis +1 %", "+1 bis +3 %", "> +3 %"]
    df["gap_bucket"] = pd.cut(df["gap"], [-np.inf, -0.03, -0.01, -0.0025, 0.0025, 0.01, 0.03, np.inf], labels=gap_labels).astype(str)

    df["loc10"] = np.select(
        [df["p_or"] > df["prev_h"], df["p_or"] < df["prev_l"]],
        ["über Vortageshoch", "unter Vortagestief"], "dazwischen")
    loc_labels = ["unter Vortagestief", "dazwischen", "über Vortageshoch"]

    orb = pd.concat([
        df.assign(orb="Ausbruch nach oben", r_orb=df["r_long"], spy_orb=df["spy_long"]).dropna(subset=["long_entry"]),
        df.assign(orb="Ausbruch nach unten", r_orb=df["r_short"], spy_orb=df["spy_short"]).dropna(subset=["short_entry"]),
    ])
    orb_labels = ["Ausbruch nach oben", "Ausbruch nach unten"]

    analyses = [
        ("1 · Eröffnungslücke: weiterlaufen oder schließen?",
         "Rendite von der Eröffnung bis zum Schluss, gruppiert nach der Lücke zum Vortagesschluss. "
         "Positiv = Kurs steigt nach der Eröffnung weiter, negativ = fällt.",
         analyse(df, "gap_bucket", "r_day", "spy_day", gap_labels, cost),
         "Die graue Spalte zeigt die rohe Rendite ohne Abzug der Marktbewegung."),
        ("2 · Lage um 10:00 zum Vortageshoch und -tief",
         "Rendite von 10:00 (Eröffnung der 10:00-Kerze) bis Schluss, je nachdem, ob der Kurs um 10:00 über dem Vortageshoch, unter dem Vortagestief oder dazwischen liegt.",
         analyse(df, "loc10", "r_rest", "spy_rest", loc_labels, cost),
         "„Dazwischen“ ist der Normalfall und dient als Vergleich."),
        ("3 · Opening Range Breakout (erste 30 Minuten)",
         "Signal: erster 5-Min-Schlusskurs nach 10:00 über dem Hoch (long) bzw. unter dem Tief (short) der ersten 30 Minuten. Einstieg zur Eröffnung der folgenden Kerze, Ausstieg zum Tagesschluss. "
         "Positiv = Kurs steigt nach dem Signal (Ausbruch läuft weiter bei „oben“, kehrt um bei „unten“).",
         analyse(orb, "orb", "r_orb", "spy_orb", orb_labels, cost),
         "Wichtig: Roh (graue Spalte) sieht der Ausbruch nach unten stark aus. Der Großteil davon ist aber die Erholung des Gesamtmarkts – marktbereinigt bleibt deutlich weniger. "
         "Ohne Stop-Loss und ohne Filter, also die einfachste Form. Die Spalte „Test ab 23“ zeigt in Klammern, in welche Richtung gehandelt würde. An einem Tag können beide Signale auslösen."),
    ]
    out = data_dir / "analysis.html"
    out.write_text(render(df, analyses, cost), encoding="utf-8")
    print(f"Seite geschrieben: {out}")
    for title, _, res, _ in analyses:
        print(f"\n{title}")
        for r in res["rows"]:
            print(f"  {r['group']:<28} 16–22: {r['is']['mean']:+6.1f} bps (t {r['is']['t']:+5.1f}) | "
                  f"ab 23: {r['oos']['mean']:+6.1f} bps (t {r['oos']['t']:+5.1f}) | netto {r['oos_net']:+6.1f} | "
                  f"{'ROBUST' if r['robust'] else '-'}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
