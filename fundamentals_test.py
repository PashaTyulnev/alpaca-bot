"""Wie gut sagen Fundamentaldaten (SEC, zeitpunktgenau) künftige Renditen voraus? -> data/fundamentals.html

Je Monatsende werden aus den bis dahin VERÖFFENTLICHTEN Jahreszahlen Kennzahlen gebildet und mit der Rendite der
folgenden 1, 3 und 12 Monate verglichen (Rangkorrelation „IC“). Dazu Portfolio-Backtests (oberstes vs. unterstes
Zehntel, monatlich umgeschichtet) mit dem Gerüst aus backtest.py.

Die Richtung jeder Kennzahl ist VORAB aus der Forschung festgelegt (z. B. günstig bewertet = gut). Getestet wird
also, ob bekannte Effekte in diesen Daten auftreten – nicht, ob sich irgendein Zusammenhang finden lässt.
Entwicklungszeitraum bis 2017, Prüfzeitraum ab 2018.

Voraussetzung: python sec_fundamentals.py
Start:  python fundamentals_test.py
"""

import argparse
import html
import webbrowser
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import backtest as bt
from download_data import PROJECT_DIR, load_config, resolve_data_dir
from strategies_daily import rank_weights

SPLIT = pd.Timestamp("2018-01-01")
START = pd.Timestamp("2010-01-01")
HORIZONS = (1, 3, 12)

# Kennzahl: (Beschreibung, erwartete Richtung laut Forschung: +1 = hoher Wert gut, −1 = niedriger Wert gut)
FACTORS = {
    "ep": ("Gewinnrendite (Gewinn / Börsenwert, Kehrwert des KGV)", +1),
    "bm": ("Buchwert / Börsenwert (Kehrwert des KBV)", +1),
    "fcfy": ("Free-Cashflow-Rendite ((Cashflow − Investitionen) / Börsenwert)", +1),
    "sp": ("Umsatz / Börsenwert (Kehrwert des KUV)", +1),
    "roe": ("Eigenkapitalrendite (Gewinn / Eigenkapital)", +1),
    "gpa": ("Bruttogewinn / Vermögen (Profitabilität nach Novy-Marx)", +1),
    "rev_growth": ("Umsatzwachstum zum Vorjahr", -1),
    "asset_growth": ("Vermögenswachstum zum Vorjahr", -1),
    "accruals": ("Bilanzielle Abgrenzungen ((Gewinn − Cashflow) / Vermögen)", -1),
    "issuance": ("Neue Aktien in 12 Monaten (Verwässerung)", -1),
    "size": ("Börsenwert (Größe)", -1),
}


def load_inputs(data_dir: Path):
    sec = data_dir / "sec"
    annual = pd.read_parquet(sec / "annual.parquet")
    shares = pd.read_parquet(sec / "shares.parquet")
    tickers = pd.read_parquet(sec / "tickers.parquet")
    splits = duckdb.connect().execute(
        "SELECT ticker, date, stock_splits FROM read_parquet(?, union_by_name=true) WHERE stock_splits > 0",
        [str(data_dir / "prices" / "1d" / "*.parquet")]).df()
    splits["date"] = pd.to_datetime(splits["date"])
    # Einheitliche Zeitauflösung, sonst verweigert merge_asof den Abgleich
    for df, cols in ((annual, ["end", "available"]), (shares, ["shares_date", "filed"]), (splits, ["date"])):
        df[cols] = df[cols].astype("datetime64[ns]")
    return annual, shares, tickers, splits


def clean_annual(annual: pd.DataFrame) -> pd.DataFrame:
    """Wachstumsraten ergänzen und veraltete Nachzügler entfernen: Ein Geschäftsjahr, das erst veröffentlicht wird,
    wenn schon ein jüngeres bekannt ist, wäre zu diesem Zeitpunkt nicht mehr die aktuellste Information."""
    a = annual.sort_values(["cik", "end"]).copy()
    prev = a.groupby("cik").shift(1)
    one_year = (a["end"] - prev["end"]).dt.days.between(330, 400)
    a["rev_growth"] = np.where(one_year, a["revenue"] / prev["revenue"].where(prev["revenue"] > 0) - 1, np.nan)
    a["asset_growth"] = np.where(one_year, a["assets"] / prev["assets"].where(prev["assets"] > 0) - 1, np.nan)
    a["available"] = np.where(one_year, np.maximum(a["available"], prev["available"].fillna(a["available"])), a["available"])
    a["available"] = pd.to_datetime(a["available"])
    a = a.sort_values(["cik", "available", "end"])
    newest_end = a.groupby("cik")["end"].cummax()
    return a[a["end"] >= newest_end]


def split_factor_after(splits: pd.DataFrame, ticker: str, dates: pd.Series) -> np.ndarray:
    """Produkt aller Splits NACH dem jeweiligen Datum: rechnet damalige Aktienzahlen auf heutige Basis um."""
    s = splits[splits["ticker"] == ticker].sort_values("date")
    if s.empty:
        return np.ones(len(dates))
    cum = np.cumprod(s["stock_splits"].to_numpy())
    total = cum[-1]
    idx = np.searchsorted(s["date"].to_numpy(), dates.to_numpy(), side="right")
    before = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 1.0)
    return total / before


def factor_panel(panel: bt.Panel, annual, shares, tickers, splits, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Lange Tabelle (Monatsende, Ticker) mit allen Kennzahlen – nur aus Daten, die VOR dem Monatsende veröffentlicht waren."""
    a = clean_annual(annual)
    out = []
    for ticker, cik in tickers[["ticker", "cik"]].itertuples(index=False):
        if ticker not in panel.raw_close.columns:
            continue
        grid = pd.DataFrame({"date": month_ends})
        fa = a[a["cik"] == cik].sort_values("available")
        if fa.empty:
            continue
        # Veröffentlichung am Monatsende-Tag selbst zählt noch nicht (konservativ: strikt davor)
        g = pd.merge_asof(grid, fa.assign(available=fa["available"] + pd.Timedelta(days=1)),
                          left_on="date", right_on="available", direction="backward")
        sh = shares[shares["cik"] == cik].sort_values("filed")
        if not sh.empty:
            sh = sh.assign(shares_adj=sh["shares"].to_numpy() * split_factor_after(splits, ticker, sh["shares_date"]),
                           filed=sh["filed"] + pd.Timedelta(days=1))
            g = pd.merge_asof(g, sh[["filed", "shares_adj"]], left_on="date", right_on="filed", direction="backward")
            g["shares_adj_12m"] = pd.merge_asof(grid.assign(date=grid["date"] - pd.DateOffset(months=12)),
                                                sh[["filed", "shares_adj"]], left_on="date", right_on="filed",
                                                direction="backward")["shares_adj"].to_numpy()
        else:
            g["shares_adj"] = g["shares_adj_12m"] = np.nan
        price = panel.raw_close[ticker].reindex(month_ends).to_numpy()
        g["mcap"] = price * g["shares_adj"]
        g["ticker"] = ticker
        out.append(g)
    f = pd.concat(out, ignore_index=True)
    m = f["mcap"].where(f["mcap"] > 0)
    eq = f["equity"].where(f["equity"] > 0)
    f["ep"] = f["net_income"] / m
    f["bm"] = eq / m
    f["fcfy"] = (f["ocf"] - f["capex"].fillna(0)) / m
    f["sp"] = f["revenue"] / m
    f["roe"] = f["net_income"] / eq
    f["gpa"] = f["gross_profit"] / f["assets"].where(f["assets"] > 0)
    f["accruals"] = (f["net_income"] - f["ocf"]) / f["assets"].where(f["assets"] > 0)
    f["issuance"] = f["shares_adj"] / f["shares_adj_12m"] - 1
    f["size"] = np.log(m)
    # Zu alte Jahreszahlen (> 18 Monate seit Geschäftsjahresende) nicht verwenden
    stale = (f["date"] - f["end"]).dt.days > 548
    f.loc[stale, list(FACTORS)] = np.nan
    return f.set_index(["date", "ticker"])[list(FACTORS) + ["mcap"]]


def forward_returns(panel: bt.Panel, month_ends: pd.DatetimeIndex) -> dict[int, pd.DataFrame]:
    px = panel.close.reindex(month_ends)
    return {h: (px.shift(-h) / px - 1).clip(-0.95, 5) for h in HORIZONS}


def ic_series(score: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Rangkorrelation zwischen Kennzahl und künftiger Rendite je Monat (Spearman)."""
    s, r = score.rank(axis=1), fwd.where(score.notna()).rank(axis=1)
    s = s.where(r.notna())
    sc, rc = s.sub(s.mean(axis=1), axis=0), r.sub(r.mean(axis=1), axis=0)
    ic = (sc * rc).sum(axis=1) / np.sqrt((sc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    return ic[s.notna().sum(axis=1) >= 100]


def ic_stats(ic: pd.Series, h: int) -> dict:
    ic = ic.dropna()
    if len(ic) < 12:
        return {"mean": np.nan, "t": np.nan}
    # Überlappende Zeiträume (3 und 12 Monate) sind nicht unabhängig: effektive Anzahl = Monate / Horizont
    return {"mean": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic) / h)}


def decile_excess(score: pd.DataFrame, fwd: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Ø Rendite der nächsten 12 Monate je Zehntel (1 = laut Forschung schlechtestes, 10 = bestes), abzüglich Ø aller."""
    sc, r = score[mask.to_numpy()], fwd[mask.to_numpy()]
    dec = np.ceil(sc.rank(axis=1, pct=True) * 10)
    excess = r.sub(r.where(sc.notna()).mean(axis=1), axis=0)
    return np.array([excess.where(dec == d).stack().mean() for d in range(1, 11)])


# ---------- Seite ----------

def f2(v, d=3) -> str:
    return "–" if v is None or np.isnan(v) else f"{v:+.{d}f}".replace(".", ",").replace("-", "−")


def pct(v, d=1) -> str:
    return "–" if v is None or np.isnan(v) else f"{v * 100:+.{d}f} %".replace(".", ",").replace("-", "−")


def mini_chart(is_vals: np.ndarray, oos_vals: np.ndarray) -> str:
    w, h, pad = 260, 110, 4
    lim = max(np.nanmax(np.abs(np.concatenate([is_vals, oos_vals]))), 0.02)
    y = lambda v: h / 2 - v / lim * (h / 2 - pad)
    band = w / 10
    bar = min(10, band / 2.6)
    parts = [f'<svg viewBox="0 0 {w} {h + 14}" class="mini"><line x1="0" x2="{w}" y1="{h / 2}" y2="{h / 2}" class="zero"/>']
    for i in range(10):
        for k, (vals, cls) in enumerate(((is_vals, "s1"), (oos_vals, "s2"))):
            v = vals[i]
            if np.isnan(v):
                continue
            x = band * i + band / 2 - bar - 1 + k * (bar + 2)
            top, bottom = sorted((y(v), y(0)))
            parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar:.1f}" height="{max(bottom - top, 0.5):.1f}" rx="2" class="{cls}">'
                         f'<title>Zehntel {i + 1} · {"bis 2017" if k == 0 else "ab 2018"}: {pct(v)} p. a. ggü. Durchschnitt</title></rect>')
        parts.append(f'<text x="{band * i + band / 2:.1f}" y="{h + 11}" class="axis" text-anchor="middle">{i + 1}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(rows: list[dict], combo: dict, n_firms: int) -> str:
    table = []
    for r in rows:
        ic = r["ic"]
        cells = "".join(f"<td class='n'>{f2(ic[(p, h)]['mean'])}<span class='t'> ({f2(ic[(p, h)]['t'], 1)})</span></td>"
                        for p in ("is", "oos") for h in HORIZONS)
        ok = r["verdict"]
        table.append(f"<tr><td><b>{html.escape(r['name'])}</b><div class='muted small'>{html.escape(r['desc'])}</div></td>{cells}"
                     f"<td class='n'>{f2(r['ls_is'], 2)}</td><td class='n'><b>{f2(r['ls_oos'], 2)}</b></td>"
                     f"<td><span class='tag {'ok' if ok == 'bestätigt' else 'warn' if ok == 'nur bis 2017' else ''}'>{ok}</span></td></tr>")
    charts = "".join(
        f"<div class='mc'><div class='mct'>{html.escape(r['name'])}</div>{mini_chart(r['dec_is'], r['dec_oos'])}</div>" for r in rows + [combo])
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fundamentaldaten</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --accent:#2f6fde; --s1:#2a78d6; --s2:#eb6834; --ok:#1f8a4c; --warn:#b26a00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30; --accent:#5b8ff0; --s1:#3987e5; --s2:#d95926; --ok:#4cc27e; --warn:#e0a040; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:0 auto; }} a {{ color:var(--accent); text-decoration:none; }}
h1 {{ font-size:22px; margin:4px 0; }} h2 {{ font-size:17px; margin:0 0 8px; }} .muted {{ color:var(--muted); }} .small {{ font-size:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }}
.verdict {{ border-left:3px solid var(--accent); padding:6px 12px; }}
li {{ margin:4px 0; }}
.scroll {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-size:11px; font-weight:600; color:var(--muted); vertical-align:bottom; }} td.n {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.t {{ color:var(--muted); font-size:11px; }}
.tag {{ display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px; border:1px solid currentColor; color:var(--muted); white-space:nowrap; }}
.tag.ok {{ color:var(--ok); }} .tag.warn {{ color:var(--warn); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(250px, 100%), 1fr)); gap:14px; }}
.mct {{ font-size:12px; font-weight:600; margin-bottom:2px; }}
.mini {{ width:100%; height:auto; }} .mini .zero {{ stroke:var(--muted); stroke-width:1; }} .mini .s1 {{ fill:var(--s1); }} .mini .s2 {{ fill:var(--s2); }}
.mini .axis {{ fill:var(--muted); font-size:9px; }}
.legend {{ display:flex; gap:16px; font-size:12px; color:var(--muted); margin-bottom:10px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
</style></head><body><main>
<a href="overview.html">← Datenübersicht</a> · <a href="backtests.html">Tagesstrategien</a> · <a href="analysis.html">Eröffnungs-Analysen</a> · <a href="long_only.html">Long-only Q+V</a>
<h1>Fundamentaldaten und künftige Rendite</h1>
<p class="muted">SEC EDGAR, zeitpunktgenau · {n_firms:,} Firmen · Jahreszahlen ab ihrer ersten Veröffentlichung · Universum: 1.000 liquideste Aktien je Monat</p>
<div class="card"><b>So liest du die Ergebnisse</b><ul class="muted">
<li><b>IC</b> (Informationskoeffizient) = Rangkorrelation zwischen Kennzahl und Rendite der folgenden 1, 3 bzw. 12 Monate, jeden Monat neu über alle Aktien. Positiv = die Kennzahl zeigt in die Richtung, die die Forschung erwartet. In der Praxis gilt ein IC von +0,02 bis +0,05 schon als nützlich. In Klammern steht der t-Wert, ab etwa |2| eher kein Zufall.</li>
<li><b>Sharpe Long-Short</b> = Portfolio aus dem laut Forschung besten Zehntel minus dem schlechtesten, monatlich umgeschichtet, nach 5 bps Kosten (Gerüst aus <code>backtest.py</code>).</li>
<li>Richtungen sind vorab aus der Forschung festgelegt. Bis 2017 = Entwicklungszeitraum, ab 2018 = Prüfzeitraum.</li>
<li><b>Grenzen:</b> nur heute noch gelistete Firmen (Survivorship Bias). Das schönt besonders „billige“ Aktien, denn die Pleitefirmen fehlen. Dazu kommen nur Jahreszahlen und nur Firmen mit US-Bilanzierung (keine ausländischen IFRS-Berichte).</li>
</ul></div>
<div class="card verdict">{combo['text']}</div>
<div class="card"><h2>Kennzahlen im Vergleich</h2><div class="scroll"><table>
<tr><th rowspan="2">Kennzahl</th><th colspan="3" class="n">IC bis 2017</th><th colspan="3" class="n">IC ab 2018</th><th colspan="2" class="n">Sharpe Long-Short</th><th rowspan="2">Urteil</th></tr>
<tr><th class="n">1 M</th><th class="n">3 M</th><th class="n">12 M</th><th class="n">1 M</th><th class="n">3 M</th><th class="n">12 M</th><th class="n">bis 2017</th><th class="n">ab 2018</th></tr>
{''.join(table)}
<tr><td><b>Kombination</b><div class='muted small'>Durchschnitt aller Kennzahlen-Ränge (je Aktie mindestens 5 vorhanden)</div></td>
{''.join(f"<td class='n'>{f2(combo['ic'][(p, h)]['mean'])}<span class='t'> ({f2(combo['ic'][(p, h)]['t'], 1)})</span></td>" for p in ("is", "oos") for h in HORIZONS)}
<td class='n'>{f2(combo['ls_is'], 2)}</td><td class='n'><b>{f2(combo['ls_oos'], 2)}</b></td><td><span class='tag {'ok' if combo['verdict'] == 'bestätigt' else ''}'>{combo['verdict']}</span></td></tr>
</table></div></div>
<div class="card"><h2>Rendite je Zehntel (nächste 12 Monate, gegenüber dem Durchschnitt)</h2>
<p class="muted small">Zehntel 1 = laut Forschung schlechteste Aktien, 10 = beste. Wenn die Kennzahl funktioniert, steigen die Balken von links nach rechts.</p>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>bis 2017</span><span><i class="sw" style="background:var(--s2)"></i>ab 2018</span></div>
<div class="grid">{charts}</div></div>
<p class="muted small">Erzeugt am {datetime.now():%d.%m.%Y %H:%M} mit fundamentals_test.py</p>
</main></body></html>""".replace(f"{n_firms:,}", f"{n_firms:,}".replace(",", "."))


def verdict(ic: dict, ls_is: float, ls_oos: float) -> str:
    is_ok = ic[("is", 12)]["mean"] > 0 and ic[("is", 12)]["t"] > 2
    oos_ok = ic[("oos", 12)]["mean"] > 0 and ic[("oos", 12)]["t"] > 2 and ls_oos > 0.3
    if is_ok and oos_ok:
        return "bestätigt"
    if is_ok:
        return "nur bis 2017"
    if oos_ok:
        return "nur ab 2018"
    return "kein Effekt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fundamentaldaten gegen künftige Renditen testen.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(load_config(args.config))
    panel = bt.load_panel(data_dir, start="2008-01-01")
    annual, shares, tickers, splits = load_inputs(data_dir)
    days = panel.close.index
    month_ends = pd.DatetimeIndex(days.to_series().groupby(days.to_period("M")).max().values)
    month_ends = month_ends[month_ends >= START].astype("datetime64[ns]")

    print("Kennzahlen je Monat berechnen ...")
    fp = factor_panel(panel, annual, shares, tickers, splits, month_ends)
    fp.to_parquet(data_dir / "sec" / "factors_monthly.parquet")  # Zwischenspeicher für long_only.py
    universe = bt.liquid_universe(panel, args.top)
    uni_m = universe.reindex(month_ends).fillna(False)
    fwd = forward_returns(panel, month_ends)
    market = panel.r_cc["SPY"].fillna(0.0)
    is_mask = pd.Series(month_ends < SPLIT, index=month_ends)

    def evaluate_score(score: pd.DataFrame) -> dict:
        score = score.where(uni_m)
        ic = {}
        for h in HORIZONS:
            s = ic_series(score, fwd[h])
            ic[("is", h)] = ic_stats(s[s.index < SPLIT], h)
            ic[("oos", h)] = ic_stats(s[s.index >= SPLIT], h)
        daily_score = score.reindex(days)
        w = rank_weights(daily_score, universe, month_ends, long_short=True)
        res = bt.evaluate(bt.run(panel, w, "LS", args.cost_bps), market, str(SPLIT.date()))
        return {"ic": ic, "ls_is": res.stats["is"].get("sharpe", np.nan), "ls_oos": res.stats["oos"].get("sharpe", np.nan),
                "dec_is": decile_excess(score, fwd[12], is_mask), "dec_oos": decile_excess(score, fwd[12], ~is_mask)}

    rows, ranks = [], []
    for key, (desc, sign) in FACTORS.items():
        score = (fp[key].unstack("ticker") * sign).reindex(index=month_ends, columns=panel.close.columns)
        ranks.append(score.where(uni_m).rank(axis=1, pct=True))
        r = evaluate_score(score)
        r.update(name=key, desc=desc, verdict=verdict(r["ic"], r["ls_is"], r["ls_oos"]))
        rows.append(r)
        print(f"{key:<13} IC12 bis 2017 {r['ic'][('is', 12)]['mean']:+.3f} (t {r['ic'][('is', 12)]['t']:+.1f}) | "
              f"ab 2018 {r['ic'][('oos', 12)]['mean']:+.3f} (t {r['ic'][('oos', 12)]['t']:+.1f}) | "
              f"Sharpe LS {r['ls_is']:+.2f} / {r['ls_oos']:+.2f} | {r['verdict']}")

    stacked = np.stack([r.to_numpy() for r in ranks])
    enough = (~np.isnan(stacked)).sum(axis=0) >= 5
    combo_score = pd.DataFrame(np.where(enough, np.nanmean(stacked, axis=0), np.nan), index=month_ends, columns=panel.close.columns)
    combo = evaluate_score(combo_score)
    combo.update(name="Kombination", verdict=verdict(combo["ic"], combo["ls_is"], combo["ls_oos"]))
    n_ok = sum(r["verdict"] == "bestätigt" for r in rows)
    combo["text"] = (f"<b>Ergebnis:</b> {n_ok} von {len(rows)} Kennzahlen zeigen in beiden Zeiträumen den erwarteten Zusammenhang mit der "
                     f"Rendite der nächsten 12 Monate. Die Kombination aller Kennzahlen erreicht ab 2018 einen IC von "
                     f"{f2(combo['ic'][('oos', 12)]['mean'])} und als Long-Short-Portfolio eine Sharpe-Ratio von {f2(combo['ls_oos'], 2)} "
                     f"(bis 2017: {f2(combo['ls_is'], 2)}). Zum Vergleich: SPY kaufen und halten liegt ab 2018 bei etwa 0,8.")
    print(f"Kombination   IC12 bis 2017 {combo['ic'][('is', 12)]['mean']:+.3f} | ab 2018 {combo['ic'][('oos', 12)]['mean']:+.3f} | "
          f"Sharpe LS {combo['ls_is']:+.2f} / {combo['ls_oos']:+.2f}")

    out = data_dir / "fundamentals.html"
    out.write_text(render(rows, combo, int(annual["cik"].nunique())), encoding="utf-8")
    print(f"Seite geschrieben: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
