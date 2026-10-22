"""Klassische Tagesstrategien auf dem Backtest-Gerüst (backtest.py) und Ergebnisseite data/backtests.html.

Alle Parameter sind vorab festgelegt (Standardwerte aus der Forschung) und werden NICHT optimiert.
In-Sample bis 2015, Out-of-Sample ab 2016.

Start:  python strategies_daily.py
        python strategies_daily.py --cost-bps 10 --top 500
"""

import argparse
import html
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from download_data import PROJECT_DIR, load_config, resolve_data_dir

SPLIT = "2016-01-01"
DECILE = 0.10


def rank_weights(score: pd.DataFrame, universe: pd.DataFrame, rebalance: pd.Index, long_short: bool) -> pd.DataFrame:
    """Oberstes Dezil kaufen (Summe +1), bei Long-Short unterstes Dezil leerverkaufen (Summe −1).
    Gewichte werden nur an Umschichtungstagen gesetzt; dazwischen laufen die Bestände mit den Kursen mit."""
    s = score.where(universe)
    pct = s.rank(axis=1, pct=True)
    n = s.notna().sum(axis=1)
    top = (pct > 1 - DECILE) & (n >= 50).to_numpy()[:, None]
    w = top.astype(float).div(top.sum(axis=1).replace(0, np.nan), axis=0)
    if long_short:
        bottom = (pct <= DECILE) & (n >= 50).to_numpy()[:, None]
        w = w - bottom.astype(float).div(bottom.sum(axis=1).replace(0, np.nan), axis=0)
    return bt.only_on(w.fillna(0.0), rebalance)


def momentum(panel: bt.Panel, universe: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    """Rendite von vor 12 Monaten bis vor 1 Monat; der letzte Monat wird übersprungen (dort wirkt eher Umkehr)."""
    score = panel.close.shift(21) / panel.close.shift(252) - 1
    month_end = panel.close.index.to_series().groupby(panel.close.index.to_period("M")).max()
    return rank_weights(score, universe, pd.Index(month_end.values), long_short)


def reversal(panel: bt.Panel, universe: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    """Verlierer der letzten 5 Handelstage kaufen (bei Long-Short: Gewinner leerverkaufen), wöchentlich."""
    score = -(panel.close / panel.close.shift(5) - 1)
    rebalance = panel.close.index[::5]
    return rank_weights(score, universe, rebalance, long_short)


def gap_drift(panel: bt.Panel, universe: pd.DataFrame, long_short: bool, hold: int = 20) -> pd.DataFrame:
    """Ersatz für Quartalszahlen: Kurslücke ≥ 4 % bei mindestens 3-fachem Durchschnittsvolumen.
    Danach 20 Handelstage in Richtung der Lücke halten. Alle offenen Ereignisse gleich gewichtet."""
    gap = panel.open / panel.close.shift(1) - 1
    avg_vol = panel.volume.rolling(60, min_periods=40).mean().shift(1)
    event = (gap.abs() >= 0.04) & (panel.volume >= 3 * avg_vol) & universe
    up = (event & (gap > 0)).astype(float)
    down = (event & (gap < 0)).astype(float)
    # Ereignis am Tag t (nach Schluss bekannt) -> gehalten an den folgenden `hold` Tagen
    active_up = up.rolling(hold, min_periods=1).sum().clip(upper=1)
    w = active_up.div(active_up.sum(axis=1).replace(0, np.nan), axis=0)
    if long_short:
        active_down = down.rolling(hold, min_periods=1).sum().clip(upper=1)
        w = w.fillna(0.0) - active_down.div(active_down.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.fillna(0.0)


STRATEGIES = [
    ("Momentum 12-1", momentum,
     "Kauft die Aktien mit der besten Rendite der letzten 12 Monate (ohne den letzten Monat), monatlich umgeschichtet. "
     "Einer der am besten dokumentierten Effekte, aber mit seltenen, sehr heftigen Einbrüchen (z. B. 2009)."),
    ("Kurzfristige Umkehr (5 Tage)", reversal,
     "Kauft die größten Verlierer der letzten Woche, wöchentlich umgeschichtet. Viel Umschlag – Kosten entscheiden."),
    ("Drift nach Kurslücke", gap_drift,
     "Nach einer Kurslücke ≥ 4 % mit mindestens 3-fachem Volumen (meist Quartalszahlen oder News) 20 Tage in Richtung der Lücke halten."),
]


# ---------- Seite ----------

def pct(v, digits=1) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{digits}f} %".replace(".", ",").replace("-", "−")


def num(v, digits=2) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}".replace(".", ",").replace("-", "−")


def stats_table(results: list[bt.Result]) -> str:
    head = ("<tr><th>Variante</th><th>Zeitraum</th><th class='n'>Rendite p.a.</th><th class='n'>Schwankung</th>"
            "<th class='n'>Sharpe</th><th class='n'>Max. Verlust</th><th class='n'>Beta</th><th class='n'>Alpha p.a.</th>"
            "<th class='n'>Umschlag/Jahr</th></tr>")
    body = []
    for r in results:
        for key, label in (("is", "bis 2015"), ("oos", "ab 2016")):
            s = r.stats.get(key) or {}
            body.append(
                f"<tr class='{key}'><td>{html.escape(r.name) if key == 'is' else ''}</td><td class='muted'>{label}</td>"
                f"<td class='n'>{pct(s.get('cagr'))}</td><td class='n'>{pct(s.get('vol'), 0)}</td>"
                f"<td class='n'><b>{num(s.get('sharpe'))}</b></td><td class='n'>{pct(s.get('maxdd'), 0)}</td>"
                f"<td class='n'>{num(s.get('beta'))}</td><td class='n'>{pct(s.get('alpha'))}</td>"
                f"<td class='n'>{num(s.get('turnover_year'), 1) if s.get('turnover_year') else '–'}</td></tr>"
            )
    return f"<table>{head}{''.join(body)}</table>"


def years_table(results: list[bt.Result]) -> str:
    years = sorted({y for r in results for y in r.stats["years"].index})
    head = "<tr><th>Variante</th>" + "".join(f"<th class='n'>{y}</th>" for y in years) + "</tr>"
    body = []
    for r in results:
        cells = []
        for y in years:
            v = r.stats["years"].get(y, np.nan)
            if np.isnan(v):
                cells.append("<td class='n muted'>–</td>")
                continue
            alpha = min(abs(v) / 0.4, 1) * 0.5
            rgb = "var(--pos-rgb)" if v > 0 else "var(--neg-rgb)"
            cells.append(f"<td class='n' style='background:rgba({rgb},{alpha:.2f})'>{pct(v, 0)}</td>")
        body.append(f"<tr><td>{html.escape(r.name)}</td>{''.join(cells)}</tr>")
    return f"<table class='years'>{head}{''.join(body)}</table>"


def chart_data(results: list[bt.Result]) -> list[dict]:
    """Monatsend-Werte des Kapitalverlaufs (Start = 1)."""
    out = []
    for r in results:
        eq = (1 + r.daily["ret"]).cumprod()
        m = eq.resample("ME").last().dropna()
        out.append({"name": r.name, "x": [d.strftime("%Y-%m") for d in m.index], "y": [round(float(v), 4) for v in m.values]})
    return out


def verdict(ls: bt.Result) -> str:
    i, o = ls.stats.get("is") or {}, ls.stats.get("oos") or {}
    si, so = i.get("sharpe", np.nan), o.get("sharpe", np.nan)
    if np.isnan(so):
        return "Zu wenige Daten."
    if si > 0.3 and so > 0.3:
        return (f"Long-Short (reines Signal) in beiden Zeiträumen positiv (Sharpe {num(si)} / {num(so)}). "
                "Kandidat für einen genaueren Test – Leihgebühren für Leerverkäufe und Survivorship Bias sind noch nicht berücksichtigt.")
    if si > 0.3 and so <= 0.3:
        return f"Bis 2015 positiv (Sharpe {num(si)}), ab 2016 nicht mehr ({num(so)}). Der Effekt scheint verschwunden oder war Zufall."
    return f"Das reine Signal (Long-Short) zeigt keinen belastbaren Vorteil (Sharpe {num(si)} / {num(so)})."


def render(sections: list[dict], benchmarks: list[bt.Result], cost: float, top: int) -> str:
    cards = []
    for i, sec in enumerate(sections):
        series = sec["results"] + benchmarks
        cards.append(f"""<section class="card">
<h2>{html.escape(sec['title'])}</h2><p class="muted">{html.escape(sec['desc'])}</p>
<p class="verdict">{html.escape(verdict(sec['results'][1]))}</p>
<div class="chart" data-idx="{i}"></div>
<div class="scroll">{stats_table(series)}</div>
<details><summary>Rendite je Kalenderjahr</summary><div class="scroll">{years_table(series)}</div></details>
</section>""")
    data = json.dumps([chart_data(sec["results"] + benchmarks) for sec in sections]).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tagesstrategien</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --grid:#ececea; --accent:#2f6fde;
        --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --pos-rgb:31,138,76; --neg-rgb:227,73,72; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30; --grid:#26262a; --accent:#5b8ff0;
          --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --pos-rgb:76,194,126; --neg-rgb:230,103,103; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; }}
a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
h1 {{ font-size:22px; margin:4px 0; }} h2 {{ font-size:17px; margin:0 0 4px; }}
.muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }}
.verdict {{ border-left:3px solid var(--accent); padding:4px 10px; margin:10px 0; }}
.intro li {{ margin:4px 0; }}
.chart {{ position:relative; margin:8px 0; }}
.chart svg {{ width:100%; height:auto; display:block; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }} .chart .axis {{ fill:var(--muted); font-size:11px; }}
.chart .lbl {{ font-size:11px; fill:var(--fg); }}
.chart .hair {{ stroke:var(--muted); stroke-width:1; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); }}
.legend i {{ display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:3px; }}
.tip {{ position:absolute; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:6px 10px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.15); display:none; white-space:nowrap; }}
.tip i {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
th, td {{ padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ font-size:11px; font-weight:600; color:var(--muted); }}
tr.is td {{ border-bottom:0; }}
.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--muted); }}
footer {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style></head>
<body><main>
<a href="overview.html">← Datenübersicht</a> · <a href="analysis.html">Eröffnungs-Analysen</a> · <a href="correlations.html">Korrelationen</a> · <a href="breadth.html">Marktbreite</a> · <a href="fundamentals.html">Fundamentaldaten</a>
<h1>Tagesstrategien</h1>
<p class="muted">Backtest auf Tagesdaten (Yahoo), Universum: die {top} liquidesten Aktien des jeweiligen Tages · Kosten {cost:.0f} bps je Kauf/Verkauf</p>
<div class="card intro"><b>So liest du die Ergebnisse</b><ul class="muted">
<li><b>Long</b> = nur kaufen. <b>Long-Short</b> = die besten kaufen und die schlechtesten leerverkaufen. Das ist das reine Signal ohne Marktrichtung und das eigentliche Urteil.</li>
<li><b>Universum gleichgewichtet</b> = alle Aktien des Universums zu gleichen Teilen. Es enthält denselben Survivorship Bias wie die Strategien. Eine Long-Strategie ist nur dann gut, wenn sie <i>dieses</i> Portfolio schlägt, nicht nur SPY.</li>
<li><b>Sharpe</b> = Rendite je Risiko. SPY liegt langfristig bei etwa 0,5–0,7. Ab etwa 1 wäre es bemerkenswert. <b>Alpha</b> = Rendite, die nicht vom Mitschwingen mit dem Markt kommt.</li>
<li>Parameter sind vorab festgelegt (Standardwerte aus der Forschung) und nicht optimiert. Bis 2015 = Entwicklungszeitraum, ab 2016 = Prüfzeitraum.</li>
<li>Signal nach Börsenschluss, Kauf zur Eröffnung am nächsten Tag. Leihgebühren für Leerverkäufe und Steuern sind nicht enthalten.</li>
</ul></div>
{''.join(cards)}
<footer>Erzeugt am {datetime.now():%d.%m.%Y %H:%M} mit strategies_daily.py · Tagesrenditen je Aktie auf {bt.RET_CLIP[0]:.0%} / +{bt.RET_CLIP[1]:.0%} gekappt (Datenfehler).</footer>
</main>
<div id="data" hidden>{html.escape(data)}</div>
<script>
const ALL = JSON.parse(document.getElementById("data").textContent);
const COLORS = ["--c1", "--c2", "--c3", "--c4"].map(v => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const fmt = v => (v >= 10 ? v.toFixed(0) : v.toFixed(2)).replace(".", ",");
document.querySelectorAll(".chart").forEach(box => {{
  const S = ALL[+box.dataset.idx], W = 800, H = 300, L = 46, R = 200, T = 10, B = 26;
  const xs = [...new Set(S.flatMap(s => s.x))].sort(), xi = new Map(xs.map((x, i) => [x, i]));
  const vals = S.flatMap(s => s.y).filter(v => v > 0), lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
  const X = i => L + i / (xs.length - 1) * (W - L - R), Y = v => T + (hi - Math.log10(v)) / (hi - lo) * (H - T - B);
  const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`); svg.setAttribute("role", "img");
  const el = (tag, attrs, text) => {{ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; svg.append(e); return e; }};
  for (const v of [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]) {{
    if (Math.log10(v) < lo - 0.05 || Math.log10(v) > hi + 0.05) continue;
    el("line", {{ x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" }});
    el("text", {{ x: L - 6, y: Y(v) + 4, class: "axis", "text-anchor": "end" }}, "×" + String(v).replace(".", ","));
  }}
  for (let i = 0; i < xs.length; i++) if (xs[i].endsWith("-01") && +xs[i].slice(0, 4) % 4 === 0)
    el("text", {{ x: X(i), y: H - 8, class: "axis", "text-anchor": "middle" }}, xs[i].slice(0, 4));
  const split = xi.get("2016-01");
  if (split !== undefined) {{ el("line", {{ x1: X(split), x2: X(split), y1: T, y2: H - B, class: "hair" }}); el("text", {{ x: X(split) + 4, y: T + 10, class: "axis" }}, "Prüfzeitraum →"); }}
  const ends = [];
  S.forEach((s, k) => {{
    const d = s.x.map((x, j) => (j ? "L" : "M") + X(xi.get(x)).toFixed(1) + "," + Y(Math.max(s.y[j], 1e-3)).toFixed(1)).join("");
    el("path", {{ d, fill: "none", stroke: COLORS[k], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }});
    ends.push({{ k, y: Y(s.y[s.y.length - 1]), label: s.name + " ×" + fmt(s.y[s.y.length - 1]) }});
  }});
  ends.sort((a, b) => a.y - b.y); for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  ends.forEach(e => {{ el("circle", {{ cx: W - R + 6, cy: e.y - 4, r: 4, fill: COLORS[e.k] }}); el("text", {{ x: W - R + 14, y: e.y, class: "lbl" }}, e.label); }});
  const hair = el("line", {{ y1: T, y2: H - B, class: "hair", visibility: "hidden" }});
  box.append(svg);
  const legend = document.createElement("div"); legend.className = "legend";
  S.forEach((s, k) => {{ const sp = document.createElement("span"); const i = document.createElement("i"); i.style.background = COLORS[k]; sp.append(i, document.createTextNode(s.name)); legend.append(sp); }});
  box.append(legend);
  const tip = document.createElement("div"); tip.className = "tip"; box.append(tip);
  svg.addEventListener("pointermove", ev => {{
    const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(xs.length - 1, Math.round((x - L) / (W - L - R) * (xs.length - 1))));
    hair.setAttribute("x1", X(i)); hair.setAttribute("x2", X(i)); hair.setAttribute("visibility", "visible");
    tip.replaceChildren(); const h = document.createElement("div"); h.className = "muted"; h.textContent = xs[i]; tip.append(h);
    S.forEach((s, k) => {{ const j = s.x.indexOf(xs[i]); if (j < 0) return; const row = document.createElement("div"); const dot = document.createElement("i"); dot.style.background = COLORS[k];
      const b = document.createElement("b"); b.textContent = "×" + fmt(s.y[j]) + " "; row.append(dot, b, document.createTextNode(s.name)); tip.append(row); }});
    tip.style.display = "block"; const bx = box.getBoundingClientRect();
    tip.style.left = Math.min(ev.clientX - bx.left + 12, bx.width - tip.offsetWidth - 4) + "px"; tip.style.top = (ev.clientY - bx.top + 12) + "px";
  }});
  svg.addEventListener("pointerleave", () => {{ tip.style.display = "none"; hair.setAttribute("visibility", "hidden"); }});
}});
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Tagesstrategien backtesten.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Kosten je Kauf/Verkauf in bps (Standard 5)")
    parser.add_argument("--top", type=int, default=1000, help="Größe des Universums (Standard 1000)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(load_config(args.config))
    panel = bt.load_panel(data_dir)
    universe = bt.liquid_universe(panel, args.top)
    market = panel.r_cc["SPY"].fillna(0.0)

    spy = bt.evaluate(bt.buy_and_hold(panel, "SPY", "SPY kaufen & halten"), market, SPLIT)
    ew = bt.evaluate(bt.run(panel, bt.equal_weight(universe), "Universum gleichgewichtet (monatlich)", args.cost_bps), market, SPLIT)
    sections = []
    for title, fn, desc in STRATEGIES:
        results = [bt.evaluate(bt.run(panel, fn(panel, universe, ls), f"{'Long-Short' if ls else 'Long'}", args.cost_bps), market, SPLIT)
                   for ls in (False, True)]
        sections.append({"title": title, "desc": desc, "results": results})
        for r in results:
            i, o = r.stats["is"], r.stats["oos"]
            print(f"{title:<30} {r.name:<11} bis 2015: {i['cagr']:+6.1%} p.a., Sharpe {i['sharpe']:+.2f} | "
                  f"ab 2016: {o['cagr']:+6.1%} p.a., Sharpe {o['sharpe']:+.2f}, MaxDD {o['maxdd']:.0%} | gekappt: {r.stats['clipped']}")
    for r in (spy, ew):
        print(f"{r.name:<42} bis 2015: Sharpe {r.stats['is']['sharpe']:+.2f} | ab 2016: {r.stats['oos']['cagr']:+6.1%} p.a., Sharpe {r.stats['oos']['sharpe']:+.2f}")

    out = data_dir / "backtests.html"
    out.write_text(render(sections, [ew, spy], args.cost_bps, args.top), encoding="utf-8")
    print(f"Seite geschrieben: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
