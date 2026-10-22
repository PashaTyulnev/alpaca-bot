"""Long-only mit Qualitäts- und Bewertungsfilter (SEC-Fundamentaldaten) -> data/long_only.html

=====================================================================================================
VORAB FESTGELEGT (22.10.2026, vor der ersten Berechnung – danach nicht verändert)
-----------------------------------------------------------------------------------------------------
Hauptvariante „Q+V Top 20 %“:
  * Universum: je Monat die 1.000 liquidesten Aktien (damalige Liquidität, ≥ 5 Mio. $ Tagesumsatz,
    Kurs ≥ 5 $) mit bekanntem Börsenwert.
  * Score = Durchschnitt der Perzentil-Ränge aus fünf Kennzahlen (mindestens drei vorhanden):
    Gewinnrendite, Free-Cashflow-Rendite, Eigenkapitalrendite, Bruttogewinn/Vermögen, wenig neue Aktien.
    Auswahl nach Forschung (Value + Quality), NICHT nach den eigenen Ergebnissen: Bruttogewinn/Vermögen
    bleibt drin, obwohl er in unserem Test nichts gezeigt hat.
  * Kaufen: oberste 20 % nach Score, gleichgewichtet. Umschichtung am Monatsende nach Schluss, Ausführung
    zur Eröffnung des Folgetags, dazwischen halten. Kosten 10 bps je gehandeltem Dollar (vorsichtig).
  * Zeitraum: 2012 bis heute (vorher zu wenige SEC-Daten). Teilzeiträume 2012–2017 und ab 2018.

Erfolgskriterien – ALLE müssen erfüllt sein:
  K1  Sharpe-Ratio über den Gesamtzeitraum höher als SPY kaufen & halten.
  K2  Überrendite gegenüber dem gleichgewichteten Universum: t-Wert > 2 (Monatsrenditen) UND in beiden
      Teilzeiträumen positiv.
  K3  Besser als 95 % von 200 Zufallsdepots gleicher Größe aus demselben Universum (Sharpe-Ratio).
  K4  Robust: Überrendite ggü. dem gleichgewichteten Universum in mindestens 80 % der Varianten
      (Kosten 5/20 bps, Universum 500/2.000, Top 10 %/30 %, quartalsweise).

Offenlegung: Die fünf Kennzahlen wurden zuvor einzeln auf 2012–2026 untersucht (fundamentals_test.py).
Der Zeitraum ab 2018 ist daher nicht völlig unberührt. Ein wirklich sauberer Test ist nur Paper-Trading.
=====================================================================================================

Voraussetzungen: python sec_fundamentals.py && python fundamentals_test.py   (erzeugt factors_monthly.parquet)
Start:  python long_only.py
"""

import argparse
import html
import re
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from download_data import PROJECT_DIR, load_config, resolve_data_dir

START = pd.Timestamp("2012-01-01")
SPLIT = pd.Timestamp("2018-01-01")
QV = {"ep": +1, "fcfy": +1, "roe": +1, "gpa": +1, "issuance": -1}
ALL11 = {"ep": +1, "bm": +1, "fcfy": +1, "sp": +1, "roe": +1, "gpa": +1, "rev_growth": -1, "asset_growth": -1,
         "accruals": -1, "issuance": -1, "size": -1}
MAIN = {"top": 1000, "frac": 0.20, "cost": 10.0, "freq": "M"}
N_RANDOM = 200


# ---------- Bausteine ----------

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    me = bt.month_ends(index)
    me = me[me >= START]
    return me[me.month.isin([3, 6, 9, 12])] if freq == "Q" else me


def wide(fp: pd.DataFrame, col: str, dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    return fp[col].unstack("ticker").reindex(index=dates, columns=columns)


def composite(fp: pd.DataFrame, factors: dict, eligible: pd.DataFrame, min_count: int) -> pd.DataFrame:
    ranks = [(wide(fp, k, eligible.index, eligible.columns) * sign).where(eligible).rank(axis=1, pct=True)
             for k, sign in factors.items()]
    stacked = np.stack([r.to_numpy() for r in ranks])
    enough = (~np.isnan(stacked)).sum(axis=0) >= min_count
    with np.errstate(all="ignore"):
        score = np.where(enough & eligible.to_numpy(), np.nanmean(stacked, axis=0), np.nan)
    return pd.DataFrame(score, index=eligible.index, columns=eligible.columns)


def top_ew(score: pd.DataFrame, frac: float | None = None, n: int | None = None) -> pd.DataFrame:
    rank = score.rank(axis=1, ascending=False)
    count = score.notna().sum(axis=1)
    k = (count * frac).round() if frac else pd.Series(n, index=score.index)
    pick = rank.le(k, axis=0) & score.notna()
    return pick.astype(float).div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def ew(eligible: pd.DataFrame) -> pd.DataFrame:
    e = eligible.astype(float)
    return e.div(e.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def cap_weighted(mcap: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    m = mcap.where(eligible)
    return m.div(m.sum(axis=1), axis=0).fillna(0.0)


def to_daily(weights_at_rebalance: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Gewichte an Umschichtungstagen, dazwischen leer (= halten)."""
    return weights_at_rebalance.reindex(index)


# ---------- Auswertung ----------

def monthly(ret: pd.Series) -> pd.Series:
    return ret.resample("ME").apply(lambda r: (1 + r).prod() - 1)


def excess_stats(strategy: pd.Series, bench: pd.Series) -> dict:
    s, b = monthly(strategy), monthly(bench)
    x = (s - b).dropna()
    x = x[x.index >= START]
    if len(x) < 12:
        return {}
    rel = (1 + s).cumprod() / (1 + b).cumprod()
    out = {"ann": x.mean() * 12, "te": x.std() * np.sqrt(12), "t": x.mean() / x.std() * np.sqrt(len(x)),
           "hit": (x > 0).mean(), "max_rel_dd": (rel / rel.cummax() - 1).min()}
    out["ir"] = out["ann"] / out["te"]
    for name, mask in (("is", x.index < SPLIT), ("oos", x.index >= SPLIT)):
        out[f"ann_{name}"] = x[mask].mean() * 12
        out[f"t_{name}"] = x[mask].mean() / x[mask].std() * np.sqrt(mask.sum())
    return out


def period_metrics(res: bt.Result, market: pd.Series) -> dict:
    d = res.daily.loc[res.daily.index >= START]
    return {"all": bt.metrics(d["ret"], market, d["turnover"]),
            "is": bt.metrics(d.loc[d.index < SPLIT, "ret"], market, d["turnover"]),
            "oos": bt.metrics(d.loc[d.index >= SPLIT, "ret"], market, d["turnover"]),
            "years": d["ret"].groupby(d.index.year).apply(lambda r: (1 + r).prod() - 1),
            "holdings": d["long"].mean()}


# ---------- Seite ----------

def pct(v, d=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{d}f} %".replace(".", ",").replace("-", "−")


def num(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def chart_series(results: list[bt.Result]) -> list[dict]:
    out = []
    for r in results:
        d = r.daily.loc[r.daily.index >= START, "ret"]
        eq = (1 + d).cumprod().resample("ME").last()
        out.append({"name": r.name, "x": [x.strftime("%Y-%m") for x in eq.index], "y": [round(float(v), 4) for v in eq.values]})
    return out


CHART_JS = r"""
const S = JSON.parse(document.getElementById("chartdata").textContent);
const COLORS = ["--c1", "--c2", "--c3", "--c4", "--c5"].map(v => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const fmt = v => (v >= 10 ? v.toFixed(1) : v.toFixed(2)).replace(".", ",");
const box = document.getElementById("chart"), W = 900, H = 330, L = 46, R = 300, T = 10, B = 26;
const xs = [...new Set(S.flatMap(s => s.x))].sort(), xi = new Map(xs.map((x, i) => [x, i]));
const vals = S.flatMap(s => s.y).filter(v => v > 0), lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
const X = i => L + i / (xs.length - 1) * (W - L - R), Y = v => T + (hi - Math.log10(v)) / (hi - lo) * (H - T - B);
const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const el = (tag, a, text) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (text) e.textContent = text; svg.append(e); return e; };
for (const v of [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) {
  if (Math.log10(v) < lo - 0.02 || Math.log10(v) > hi + 0.02) continue;
  el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" });
  el("text", { x: L - 6, y: Y(v) + 4, class: "axis", "text-anchor": "end" }, "×" + String(v).replace(".", ","));
}
xs.forEach((x, i) => { if (x.endsWith("-01") && +x.slice(0, 4) % 2 === 0) el("text", { x: X(i), y: H - 8, class: "axis", "text-anchor": "middle" }, x.slice(0, 4)); });
const sp = xi.get("2018-01"); if (sp !== undefined) { el("line", { x1: X(sp), x2: X(sp), y1: T, y2: H - B, class: "hair" }); el("text", { x: X(sp) + 4, y: T + 10, class: "axis" }, "ab 2018"); }
const ends = [];
S.forEach((s, k) => {
  const d = s.x.map((x, j) => (j ? "L" : "M") + X(xi.get(x)).toFixed(1) + "," + Y(s.y[j]).toFixed(1)).join("");
  el("path", { d, fill: "none", stroke: COLORS[k], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" });
  ends.push({ k, y: Y(s.y[s.y.length - 1]), label: s.name + " ×" + fmt(s.y[s.y.length - 1]) });
});
ends.sort((a, b) => a.y - b.y); for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
ends.forEach(e => { el("circle", { cx: W - R + 6, cy: e.y - 4, r: 4, fill: COLORS[e.k] }); el("text", { x: W - R + 14, y: e.y, class: "lbl" }, e.label); });
const hair = el("line", { y1: T, y2: H - B, class: "hair", visibility: "hidden" });
box.append(svg);
const legend = document.createElement("div"); legend.className = "legend";
S.forEach((s, k) => { const sp = document.createElement("span"); const i = document.createElement("i"); i.style.background = COLORS[k]; sp.append(i, document.createTextNode(s.name)); legend.append(sp); });
box.append(legend);
const tip = document.createElement("div"); tip.className = "tip"; box.append(tip);
svg.addEventListener("pointermove", ev => {
  const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
  const i = Math.max(0, Math.min(xs.length - 1, Math.round((x - L) / (W - L - R) * (xs.length - 1))));
  hair.setAttribute("x1", X(i)); hair.setAttribute("x2", X(i)); hair.setAttribute("visibility", "visible");
  tip.replaceChildren(); const h = document.createElement("div"); h.className = "muted"; h.textContent = xs[i]; tip.append(h);
  S.forEach((s, k) => { const j = s.x.indexOf(xs[i]); if (j < 0) return; const row = document.createElement("div"); const dot = document.createElement("i"); dot.style.background = COLORS[k];
    const b = document.createElement("b"); b.textContent = "×" + fmt(s.y[j]) + " "; row.append(dot, b, document.createTextNode(s.name)); tip.append(row); });
  tip.style.display = "block"; const bx = box.getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - bx.left + 12, bx.width - tip.offsetWidth - 4) + "px"; tip.style.top = (ev.clientY - bx.top + 12) + "px";
});
svg.addEventListener("pointerleave", () => { tip.style.display = "none"; hair.setAttribute("visibility", "hidden"); });
"""


def render(ctx: dict) -> str:
    crit_rows = "".join(
        f"<tr><td><span class='tag {'ok' if ok else 'bad'}'>{'erfüllt' if ok else 'nicht erfüllt'}</span></td>"
        f"<td><b>{html.escape(k)}</b> {html.escape(text)}</td><td class='muted'>{html.escape(detail)}</td></tr>"
        for k, text, ok, detail in ctx["criteria"])
    perf_rows = []
    for name, m in ctx["perf"]:
        for key, label in (("all", "2012–heute"), ("is", "2012–2017"), ("oos", "ab 2018")):
            s = m[key]
            perf_rows.append(
                f"<tr class='{key}'><td>{html.escape(name) if key == 'all' else ''}</td><td class='muted'>{label}</td>"
                f"<td class='n'>{pct(s.get('cagr'))}</td><td class='n'>{pct(s.get('vol'), 0)}</td><td class='n'><b>{num(s.get('sharpe'))}</b></td>"
                f"<td class='n'>{pct(s.get('maxdd'), 0)}</td><td class='n'>{num(s.get('beta'))}</td>"
                f"<td class='n'>{num(s.get('turnover_year'), 1) if s.get('turnover_year') else '–'}</td>"
                f"<td class='n'>{num(m['holdings'], 0) if key == 'all' else ''}</td></tr>")
    ex_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td class='muted'>{html.escape(b)}</td><td class='n'><b>{pct(e.get('ann'))}</b></td>"
        f"<td class='n'>{num(e.get('t'), 1)}</td><td class='n'>{pct(e.get('ann_is'))}</td><td class='n'>{pct(e.get('ann_oos'))}</td>"
        f"<td class='n'>{pct(e.get('hit'), 0).replace('+', '')}</td><td class='n'>{pct(e.get('max_rel_dd'), 0)}</td></tr>"
        for n, b, e in ctx["excess"])
    sens_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td class='n'>{pct(m['all'].get('cagr'))}</td><td class='n'>{num(m['all'].get('sharpe'))}</td>"
        f"<td class='n'><b>{pct(e.get('ann'))}</b></td><td class='n'>{num(e.get('t'), 1)}</td>"
        f"<td class='n'>{num(m['all'].get('turnover_year'), 1)}</td></tr>" for n, m, e in ctx["sens"])
    years = sorted({y for _, m in ctx["perf"] for y in m["years"].index})
    year_rows = "".join(
        f"<tr><td>{html.escape(n)}</td>" + "".join(
            f"<td class='n'>{pct(m['years'].get(y, np.nan), 0)}</td>" for y in years) + "</tr>" for n, m in ctx["perf"])
    hold_rows = "".join(f"<tr><td><b>{html.escape(t)}</b></td><td class='muted'>{html.escape(nm)}</td><td class='n'>{num(s, 2)}</td></tr>"
                        for t, nm, s in ctx["holdings"])
    rnd = ctx["random"]
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-only Q+V</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --grid:#ececea; --accent:#2f6fde;
        --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --c5:#e87ba4; --ok:#1f8a4c; --bad:#c0392b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30; --grid:#26262a;
  --accent:#5b8ff0; --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c5:#d55181; --ok:#4cc27e; --bad:#ef6b5b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1150px; margin:0 auto; }} a {{ color:var(--accent); text-decoration:none; }}
h1 {{ font-size:22px; margin:4px 0; }} h2 {{ font-size:17px; margin:0 0 8px; }} .muted {{ color:var(--muted); }} .small {{ font-size:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }}
.verdict {{ border-left:4px solid var(--accent); font-size:15px; }}
pre {{ white-space:pre-wrap; font-size:12px; color:var(--muted); margin:0; }}
.scroll {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-size:11px; font-weight:600; color:var(--muted); white-space:nowrap; }} td.n {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
tr.all td, tr.is td {{ border-bottom:0; }}
.tag {{ display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px; border:1px solid currentColor; white-space:nowrap; }}
.tag.ok {{ color:var(--ok); }} .tag.bad {{ color:var(--bad); }}
#chart {{ position:relative; }} #chart svg {{ width:100%; height:auto; display:block; }}
#chart .grid {{ stroke:var(--grid); }} #chart .axis {{ fill:var(--muted); font-size:11px; }} #chart .lbl {{ fill:var(--fg); font-size:11px; }}
#chart .hair {{ stroke:var(--muted); }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); margin-top:6px; }}
.legend i {{ display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:3px; }}
.tip {{ position:absolute; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:6px 10px;
        font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.15); display:none; white-space:nowrap; }}
.tip i {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
details summary {{ cursor:pointer; color:var(--muted); }}
</style></head><body><main>
<a href="overview.html">← Datenübersicht</a> · <a href="fundamentals.html">Fundamentaldaten</a> · <a href="backtests.html">Tagesstrategien</a>
<h1>Long-only mit Qualitäts- und Bewertungsfilter</h1>
<p class="muted">SEC-Fundamentaldaten (zeitpunktgenau) · 2012 bis {ctx['end']:%m/%Y} · Kosten {MAIN['cost']:.0f} bps je gehandeltem Dollar · Kauf zur Eröffnung nach dem Signal</p>
<div class="card verdict">{ctx['verdict']}</div>
<div class="card"><h2>Erfolgskriterien (vorab festgelegt)</h2><div class="scroll"><table>{crit_rows}</table></div>
<details style="margin-top:10px"><summary>Vollständige Vorab-Festlegung</summary><pre>{html.escape(ctx['prereg'])}</pre></details></div>
<div class="card"><h2>Kapitalverlauf (Start 2012 = 1)</h2><div id="chart"></div></div>
<div class="card"><h2>Kennzahlen</h2><div class="scroll"><table>
<tr><th>Depot</th><th>Zeitraum</th><th class='n'>Rendite p.a.</th><th class='n'>Schwankung</th><th class='n'>Sharpe</th><th class='n'>Max. Verlust</th><th class='n'>Beta</th><th class='n'>Umschlag/Jahr</th><th class='n'>Ø Titel</th></tr>
{''.join(perf_rows)}</table></div></div>
<div class="card"><h2>Überrendite gegenüber Vergleichsdepots</h2><p class="muted small">Monatliche Differenz, annualisiert. t-Wert ab etwa 2 = eher kein Zufall. „Max. relativer Rückstand“ = größter zwischenzeitlicher Rückstand gegenüber dem Vergleich.</p><div class="scroll"><table>
<tr><th>Depot</th><th>gegen</th><th class='n'>Überrendite p.a.</th><th class='n'>t-Wert</th><th class='n'>2012–2017</th><th class='n'>ab 2018</th><th class='n'>Monate besser</th><th class='n'>Max. rel. Rückstand</th></tr>
{ex_rows}</table></div></div>
<div class="card"><h2>Zufallsdepots</h2><p class="muted">{N_RANDOM} Depots mit je {rnd['size']} zufällig gewählten Aktien aus demselben Universum, ebenfalls monatlich neu und gleichgewichtet, gleiche Kosten.
Sharpe der Hauptvariante: <b>{num(rnd['main'])}</b> · Zufallsdepots: Median {num(rnd['median'])}, 95-%-Grenze {num(rnd['p95'])} · Hauptvariante besser als <b>{rnd['pct']:.0%}</b> der Zufallsdepots.</p></div>
<div class="card"><h2>Robustheit: Varianten der Hauptvariante</h2><p class="muted small">Jeweils nur eine Einstellung geändert. Überrendite gegenüber dem gleichgewichteten Universum derselben Einstellung.</p><div class="scroll"><table>
<tr><th>Variante</th><th class='n'>Rendite p.a.</th><th class='n'>Sharpe</th><th class='n'>Überrendite p.a.</th><th class='n'>t-Wert</th><th class='n'>Umschlag/Jahr</th></tr>
{sens_rows}</table></div></div>
<div class="card"><h2>Rendite je Kalenderjahr</h2><div class="scroll"><table><tr><th>Depot</th>{''.join(f"<th class='n'>{y}</th>" for y in years)}</tr>{year_rows}</table></div></div>
<div class="card"><h2>Aktuelle Auswahl der Hauptvariante (Top 15 nach Score, Stand {ctx['asof']:%d.%m.%Y})</h2>
<p class="muted small">Nur zur Plausibilitätsprüfung der Methode – keine Kaufempfehlung.</p><div class="scroll"><table><tr><th>Ticker</th><th>Name</th><th class='n'>Score</th></tr>{hold_rows}</table></div></div>
<p class="muted small">Grenzen: nur heute noch gelistete Firmen (Survivorship Bias, schönt besonders günstig bewertete Aktien); nur Jahreszahlen; nur Firmen, deren Aktienanzahl in den SEC-Daten steht (Ø {ctx['coverage']:.0f} der 1.000 liquidesten je Monat, ausländische Firmen fehlen); Steuern nicht berücksichtigt. Erzeugt am {datetime.now():%d.%m.%Y %H:%M} mit long_only.py.</p>
</main>
<script type="application/json" id="chartdata">{json.dumps(ctx['chart']).replace('</', '<\\/')}</script>
<script>{CHART_JS}</script>
</body></html>"""


# ---------- Ablauf ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Long-only Q+V gründlich testen.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    data_dir = resolve_data_dir(load_config(args.config))

    panel = bt.load_panel(data_dir, start="2010-01-01")
    fp = pd.read_parquet(data_dir / "sec" / "factors_monthly.parquet")
    fp.index = fp.index.set_levels(pd.to_datetime(fp.index.levels[0]).astype("datetime64[ns]"), level=0)
    days = panel.close.index
    market = panel.r_cc["SPY"].fillna(0.0)
    universes = {}

    def setup(top: int, freq: str):
        """Universum, Score und Vergleichsdepots für eine Einstellung."""
        key = (top, freq)
        if key not in universes:
            reb = rebalance_dates(days, freq)
            uni = bt.liquid_universe(panel, top).reindex(reb).fillna(False)
            mcap = wide(fp, "mcap", reb, panel.close.columns)
            eligible = uni & mcap.notna()
            score = composite(fp, QV, eligible, 3)
            universes[key] = (reb, eligible, mcap, score)
        return universes[key]

    def simulate(weights: pd.DataFrame, name: str, cost: float) -> bt.Result:
        return bt.run(panel, to_daily(weights, days), name, cost)

    reb, eligible, mcap, score = setup(MAIN["top"], MAIN["freq"])
    print(f"Universum: Ø {eligible.sum(axis=1).mean():.0f} Aktien je Monat mit Börsenwert, davon Ø {score.notna().sum(axis=1).mean():.0f} mit Score")

    main_res = simulate(top_ew(score, frac=MAIN["frac"]), "Q+V Top 20 % (Hauptvariante)", MAIN["cost"])
    cw_ex = cap_weighted(mcap, eligible & ~(score.rank(axis=1, pct=True) <= 0.2))
    variants = {
        "cw_ex": simulate(cw_ex, "Kapitalgewichtet ohne schlechteste 20 %", MAIN["cost"]),
        "all11": simulate(top_ew(composite(fp, ALL11, eligible, 6), frac=MAIN["frac"]), "Alle 11 Kennzahlen Top 20 %", MAIN["cost"]),
    }
    ew_res = simulate(ew(eligible), "Universum gleichgewichtet", MAIN["cost"])
    cw_res = simulate(cap_weighted(mcap, eligible), "Universum kapitalgewichtet", MAIN["cost"])
    spy = bt.buy_and_hold(panel, "SPY", "SPY kaufen & halten")

    perf = [(r.name, period_metrics(r, market)) for r in (main_res, variants["cw_ex"], variants["all11"], ew_res, cw_res, spy)]
    for n, m in perf:
        print(f"{n:<42} {m['all']['cagr']:+6.1%} p.a. | Sharpe {m['all']['sharpe']:.2f} | MaxDD {m['all']['maxdd']:.0%} | "
              f"2012–17 Sharpe {m['is']['sharpe']:.2f} | ab 2018 Sharpe {m['oos']['sharpe']:.2f}")

    excess = [
        (main_res.name, "gleichgewichtetes Universum", excess_stats(main_res.daily["ret"], ew_res.daily["ret"])),
        (main_res.name, "SPY", excess_stats(main_res.daily["ret"], spy.daily["ret"])),
        (variants["cw_ex"].name, "kapitalgewichtetes Universum", excess_stats(variants["cw_ex"].daily["ret"], cw_res.daily["ret"])),
        (variants["cw_ex"].name, "SPY", excess_stats(variants["cw_ex"].daily["ret"], spy.daily["ret"])),
        (variants["all11"].name, "gleichgewichtetes Universum", excess_stats(variants["all11"].daily["ret"], ew_res.daily["ret"])),
    ]
    for n, b, e in excess:
        print(f"Überrendite {n} ggü. {b}: {e['ann']:+.1%} p.a., t {e['t']:+.1f}, 2012–17 {e['ann_is']:+.1%}, ab 2018 {e['ann_oos']:+.1%}")

    # Zufallsdepots gleicher Größe aus demselben Universum
    rng = np.random.default_rng(42)
    size = int(round(score.notna().sum(axis=1).median() * MAIN["frac"]))
    pool = score.notna()
    rand_sharpes = []
    for i in range(N_RANDOM):
        noise = pd.DataFrame(rng.random(pool.shape), index=pool.index, columns=pool.columns).where(pool)
        w = top_ew(noise, frac=MAIN["frac"])
        rand_sharpes.append(period_metrics(simulate(w, "zufall", MAIN["cost"]), market)["all"]["sharpe"])
        if (i + 1) % 50 == 0:
            print(f"  Zufallsdepots: {i + 1}/{N_RANDOM}")
    rand_sharpes = np.array(rand_sharpes)
    main_sharpe = perf[0][1]["all"]["sharpe"]
    random_ctx = {"size": size, "main": main_sharpe, "median": float(np.median(rand_sharpes)),
                  "p95": float(np.quantile(rand_sharpes, 0.95)), "pct": float((rand_sharpes < main_sharpe).mean())}
    print(f"Zufallsdepots: Median Sharpe {random_ctx['median']:.2f}, 95 % {random_ctx['p95']:.2f}, Hauptvariante besser als {random_ctx['pct']:.0%}")

    # Robustheit: jeweils eine Einstellung ändern
    sens = []
    settings = [("Kosten 5 bps", dict(cost=5.0)), ("Kosten 20 bps", dict(cost=20.0)),
                ("Universum 500", dict(top=500)), ("Universum 2.000", dict(top=2000)),
                ("Top 10 %", dict(frac=0.10)), ("Top 30 %", dict(frac=0.30)), ("Quartalsweise", dict(freq="Q"))]
    for label, change in settings:
        cfg = {**MAIN, **change}
        r_reb, r_elig, r_mcap, r_score = setup(cfg["top"], cfg["freq"])
        res = simulate(top_ew(r_score, frac=cfg["frac"]), label, cfg["cost"])
        bench = simulate(ew(r_elig), "ew", cfg["cost"])
        m, e = period_metrics(res, market), excess_stats(res.daily["ret"], bench.daily["ret"])
        sens.append((label, m, e))
        print(f"  {label:<16} {m['all']['cagr']:+6.1%} p.a., Sharpe {m['all']['sharpe']:.2f} | Überrendite ggü. Universum {e['ann']:+.1%} (t {e['t']:+.1f})")

    # Kriterien
    spy_sharpe = perf[-1][1]["all"]["sharpe"]
    ex_ew = excess[0][2]
    k1 = main_sharpe > spy_sharpe
    k2 = ex_ew["t"] > 2 and ex_ew["ann_is"] > 0 and ex_ew["ann_oos"] > 0
    k3 = random_ctx["pct"] >= 0.95
    share_pos = np.mean([e["ann"] > 0 for _, _, e in sens])
    k4 = share_pos >= 0.8
    criteria = [
        ("K1", "Sharpe höher als SPY", k1, f"{num(main_sharpe)} gegen {num(spy_sharpe)}"),
        ("K2", "Überrendite ggü. gleichgewichtetem Universum: t > 2 und in beiden Teilzeiträumen positiv", k2,
         f"{pct(ex_ew['ann'])} p.a., t {num(ex_ew['t'], 1)}; 2012–17 {pct(ex_ew['ann_is'])}, ab 2018 {pct(ex_ew['ann_oos'])}"),
        ("K3", "Besser als 95 % der Zufallsdepots", k3, f"besser als {random_ctx['pct']:.0%}"),
        ("K4", "Überrendite in ≥ 80 % der Varianten", k4, f"{share_pos:.0%} der Varianten"),
    ]
    n_ok = sum(c[2] for c in criteria)
    if n_ok == 4:
        verdict = ("<b>Alle vier Kriterien erfüllt.</b> Die Strategie ist ein Kandidat für Paper-Trading – nicht mehr. "
                   "Erst wenn sie dort über Monate mit neuen Daten ähnlich abschneidet, wäre echtes Geld ein Thema.")
    else:
        failed = ", ".join(c[0] for c in criteria if not c[2])
        verdict = (f"<b>{n_ok} von 4 Kriterien erfüllt – nicht bestanden</b> (verfehlt: {failed}). "
                   "Nach den vorab festgelegten Regeln ist die Strategie damit kein Kandidat für echtes Geld. "
                   "Details zu jedem Kriterium unten.")
    for c in criteria:
        print(f"{c[0]} {'OK ' if c[2] else 'NEIN'} {c[1]}: {c[3]}")

    # Aktuelle Auswahl (nur Plausibilität)
    names = {}
    uf = data_dir / "universe" / "nasdaq-common.csv"
    if uf.exists():
        names = dict(pd.read_csv(uf, dtype=str, keep_default_na=False).values)
    last = score.dropna(how="all").iloc[-1].dropna().sort_values(ascending=False).head(15)
    holdings = [(t, names.get(t, "").split(" - ")[0], float(v)) for t, v in last.items()]

    prereg = re.split(r"\n=+\n", __doc__)[1].strip()
    ctx = {"coverage": float(eligible.sum(axis=1)[eligible.sum(axis=1) > 0].mean()), "criteria": criteria, "verdict": verdict, "prereg": prereg, "perf": perf, "excess": excess, "sens": sens,
           "random": random_ctx, "holdings": holdings, "asof": score.dropna(how="all").index[-1],
           "end": main_res.daily.index[-1],
           "chart": chart_series([main_res, variants["cw_ex"], ew_res, cw_res, spy])}
    out = data_dir / "long_only.html"
    out.write_text(render(ctx), encoding="utf-8")
    print(f"Seite geschrieben: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
