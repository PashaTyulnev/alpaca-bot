"""Erzeugt eine HTML-Übersicht der lokal gespeicherten Kursdaten und öffnet sie im Browser.

Start:  python overview.py              (nutzt config.toml, schreibt data/overview.html)
        python overview.py --no-open    (nur erzeugen, nicht öffnen)
"""

import argparse
import html
import webbrowser
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from download_data import INTERVALS, PROJECT_DIR, load_config, resolve_data_dir

# Ab so vielen Kalendertagen ohne neue Tageskurse gilt ein Ticker als veraltet
STALE_AFTER_DAYS = 5


def interval_stats(folder: Path, interval: str) -> tuple[pd.DataFrame, list[str]]:
    """Zeitraum und Zeilenzahl je Ticker. Gibt zusätzlich die Namen unlesbarer Dateien zurück."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ticker", "first", "last", "rows", "dividends", "splits"]), []
    col = INTERVALS[interval]["time_column"]
    query = (
        f"SELECT ticker, min({col}) AS first, max({col}) AS last, count(*) AS rows, "
        "count(*) FILTER (WHERE dividends <> 0) AS dividends, count(*) FILTER (WHERE stock_splits <> 0) AS splits "
        "FROM read_parquet(?, union_by_name=true) GROUP BY ticker"
    )
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    try:
        return con.execute(query, [[str(f) for f in files]]).df(), []
    except duckdb.Error:
        # Mindestens eine Datei ist kaputt: einzeln lesen, um sie zu finden
        parts, broken = [], []
        for f in files:
            try:
                parts.append(con.execute(query, [[str(f)]]).df())
            except duckdb.Error:
                broken.append(f.stem)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), broken


def last_failures(log_path: Path) -> dict[tuple[str, str], str]:
    """Letzte Fehlermeldung je (Ticker, Intervall), sofern der letzte Versuch fehlschlug."""
    if not log_path.exists():
        return {}
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    latest = log.drop_duplicates(["ticker", "interval"], keep="last")
    failed = latest[latest["status"] == "fehler"]
    return {(r.ticker, r.interval): r.message for r in failed.itertuples()}


def fmt_time(value, interval: str) -> str:
    if pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    return f"{ts:%d.%m.%Y}" if interval == "1d" else f"{ts:%d.%m.%Y %H:%M}"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def render(config: dict, data_dir: Path) -> str:
    intervals = config["intervals"]
    today = date.today()

    names: dict[str, str] = {}
    universe_file = data_dir / "universe" / "nasdaq-common.csv"
    if config["universe"] == "nasdaq-common" and universe_file.exists():
        names = dict(pd.read_csv(universe_file, dtype=str, keep_default_na=False).values)
    expected = list(dict.fromkeys(config["tickers"] + list(names)))

    stats, broken, cards = {}, {}, []
    for interval in intervals:
        folder = data_dir / "prices" / interval
        df, broken[interval] = interval_stats(folder, interval)
        stats[interval] = df.set_index("ticker") if not df.empty else df
        size_mb = sum(f.stat().st_size for f in folder.glob("*.parquet")) / 1e6 if folder.exists() else 0
        cards.append(
            f'<div class="kpi"><div class="kpi-h">{interval}</div>'
            f'<div class="kpi-v">{fmt_int(len(df))} <span>Ticker</span></div>'
            f'<div class="kpi-s">{fmt_int(df["rows"].sum() if len(df) else 0)} Zeilen · {fmt_int(size_mb)} MB</div>'
            f'<div class="kpi-s">{fmt_time(df["first"].min(), interval) if len(df) else "–"} bis '
            f'{fmt_time(df["last"].max(), interval) if len(df) else "–"}</div></div>'
        )

    failures = last_failures(data_dir / "import_log.csv")
    all_tickers = expected + sorted({t for i in intervals for t in stats[i].index} - set(expected))

    d1 = stats.get("1d")
    has_1d = d1 is not None and len(d1)
    t_min = pd.Timestamp(d1["first"].min()).date() if has_1d else today
    span = max((today - t_min).days, 1)

    rows, n_problem = [], 0
    for ticker in all_tickers:
        cells, problems = [], []
        for interval in intervals:
            df = stats[interval]
            if ticker in df.index:
                r = df.loc[ticker]
                cells.append(
                    f'<td>{fmt_time(r["first"], interval)}</td><td>{fmt_time(r["last"], interval)}</td>'
                    f'<td class="n">{fmt_int(r["rows"])}</td>'
                )
            else:
                reason = failures.get((ticker, interval))
                if ticker in broken[interval]:
                    reason = "Datei nicht lesbar"
                problems.append(f"{interval}: {reason or 'noch nicht geladen'}")
                cells.append('<td colspan="3" class="miss">fehlt</td>')

        bar = ""
        if has_1d and ticker in d1.index:
            first, last = pd.Timestamp(d1.loc[ticker, "first"]).date(), pd.Timestamp(d1.loc[ticker, "last"]).date()
            if (today - last).days > STALE_AFTER_DAYS:
                problems.append(f"1d: veraltet, letzter Kurs {last:%d.%m.%Y}")
            left = (first - t_min).days / span * 100
            width = max((last - first).days / span * 100, 0.5)
            bar = f'<div class="track"><div class="fill" style="left:{left:.2f}%;width:{width:.2f}%"></div></div>'

        tags = "" if ticker in expected else '<span class="tag">nicht konfiguriert</span>'
        if problems:
            n_problem += 1
            status = f'<span class="tag warn" title="{html.escape("; ".join(problems))}">Problem</span>'
        else:
            status = '<span class="tag ok">ok</span>'
        name = html.escape(names.get(ticker, ""))
        rows.append(
            f'<tr data-p="{1 if problems else 0}"><td class="t">{html.escape(ticker)}{tags}</td>'
            f'<td class="name" title="{name}">{name}</td>{"".join(cells)}<td>{status}'
            + (f'<div class="why">{html.escape("; ".join(problems))}</div>' if problems else "")
            + f'</td><td class="bar">{bar}</td></tr>'
        )

    head_intervals = "".join(f'<th colspan="3" class="grp">{i}</th>' for i in intervals)
    sub_intervals = "".join(
        f'<th>Von</th><th>Bis</th><th class="n">{"Tage" if i == "1d" else "Kerzen"}</th>' for i in intervals
    )
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Datenübersicht</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0;
        --accent:#2f6fde; --track:#ececea; --ok:#1f8a4c; --warn:#b26a00; --bad:#c0392b; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141416; --card:#1c1c1f; --fg:#ececee; --muted:#9a9aa0; --line:#2c2c30;
          --accent:#5b8ff0; --track:#2a2a2e; --ok:#4cc27e; --warn:#e0a040; --bad:#ef6b5b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1400px; margin:0 auto; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 20px; }}
a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px; min-width:220px; }}
.kpi-h {{ font-size:12px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
.kpi-v {{ font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }}
.kpi-v span {{ font-size:13px; font-weight:400; color:var(--muted); }}
.kpi-s {{ color:var(--muted); font-size:12px; }}
.tools {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:12px; }}
.tools input[type=search] {{ flex:1; min-width:200px; max-width:360px; padding:8px 12px; border:1px solid var(--line);
               border-radius:8px; background:var(--card); color:var(--fg); font:inherit; }}
.tools label {{ color:var(--muted); display:flex; gap:6px; align-items:center; }}
#count {{ color:var(--muted); margin-left:auto; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:auto; max-height:75vh; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; vertical-align:top; }}
thead th {{ position:sticky; background:var(--card); z-index:1; font-size:12px; font-weight:600; color:var(--muted);
           text-transform:uppercase; letter-spacing:.04em; }}
thead tr:first-child th {{ top:0; }} thead tr:nth-child(2) th {{ top:33px; }}
th.grp {{ text-align:center; border-left:1px solid var(--line); }}
.t {{ font-weight:600; }}
.name {{ max-width:240px; overflow:hidden; text-overflow:ellipsis; color:var(--muted); }}
.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
.miss {{ color:var(--bad); text-align:center; }}
.why {{ color:var(--warn); font-size:12px; white-space:normal; max-width:280px; }}
.tag {{ display:inline-block; margin-left:6px; padding:1px 7px; border-radius:99px; font-size:11px;
        font-weight:500; border:1px solid currentColor; color:var(--muted); }}
td > .tag:first-child {{ margin-left:0; }}
.tag.ok {{ color:var(--ok); }} .tag.warn {{ color:var(--warn); }}
.bar {{ width:16%; min-width:140px; }}
.track {{ position:relative; height:8px; margin-top:6px; background:var(--track); border-radius:4px; }}
.fill {{ position:absolute; top:0; bottom:0; background:var(--accent); border-radius:4px; }}
footer {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style></head>
<body><main>
<h1>Datenübersicht</h1>
<p class="sub">{fmt_int(len(all_tickers))} Ticker · {fmt_int(n_problem)} mit Problemen · Universum: {html.escape(config["universe"] or "nur config.toml")}
 · <a href="correlations.html">Korrelationen →</a> · <a href="analysis.html">Eröffnungs-Analysen →</a> · <a href="backtests.html">Tagesstrategien →</a> · <a href="breadth.html">Marktbreite →</a> · <a href="fundamentals.html">Fundamentaldaten →</a> · <a href="long_only.html">Long-only Q+V →</a></p>
<div class="kpis">{"".join(cards)}</div>
<div class="tools">
  <input id="q" type="search" placeholder="Ticker oder Name suchen …">
  <label><input id="onlyp" type="checkbox"> nur Probleme</label>
  <span id="count"></span>
</div>
<div class="card"><table>
<thead><tr><th rowspan="2">Ticker</th><th rowspan="2">Name</th>{head_intervals}<th rowspan="2">Status</th>
<th rowspan="2">1d-Zeitraum ({t_min:%Y}–{today:%Y})</th></tr><tr>{sub_intervals}</tr></thead>
<tbody id="rows">
{chr(10).join(rows)}
</tbody></table></div>
<footer>Ordner: {html.escape(str(data_dir / "prices"))} · Erzeugt am {datetime.now():%d.%m.%Y %H:%M} ·
„veraltet“ = letzter Tageskurs älter als {STALE_AFTER_DAYS} Tage · Zeiten in New Yorker Börsenzeit.
Nach einem neuen Import <code>overview.py</code> erneut ausführen.</footer>
</main>
<script>
const rows = [...document.querySelectorAll("#rows tr")];
const q = document.getElementById("q"), onlyp = document.getElementById("onlyp"), count = document.getElementById("count");
function apply() {{
  const s = q.value.trim().toLowerCase();
  let n = 0;
  for (const r of rows) {{
    const show = (!s || r.textContent.toLowerCase().includes(s)) && (!onlyp.checked || r.dataset.p === "1");
    r.style.display = show ? "" : "none";
    if (show) n++;
  }}
  count.textContent = n.toLocaleString("de-DE") + " von " + rows.length.toLocaleString("de-DE") + " angezeigt";
}}
q.addEventListener("input", apply); onlyp.addEventListener("change", apply); apply();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML-Übersicht der lokalen Kursdaten erzeugen.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml", help="Pfad zur TOML-Konfiguration")
    parser.add_argument("--no-open", action="store_true", help="Seite nur erzeugen, nicht im Browser öffnen")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)

    out = data_dir / "overview.html"
    out.write_text(render(config, data_dir), encoding="utf-8")
    print(f"Übersicht geschrieben: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
