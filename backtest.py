"""Portfolio-Backtester auf Tagesdaten (Yahoo, data/prices/1d).

Grundannahmen (bewusst konservativ):
  * Eine Strategie liefert Zielgewichte je Tag und Aktie, berechnet NACH Börsenschluss dieses Tages.
  * Umgeschichtet wird zur Eröffnung des nächsten Handelstags. Über Nacht gelten noch die alten Gewichte.
  * Kosten: `cost_bps` je gehandeltem Dollar und Richtung (Spread + Slippage), Leerverkaufsgebühren nicht enthalten.
  * Universum je Tag aus der DAMALIGEN Liquidität (Dollar-Umsatz der Vortage), nicht aus der heutigen.
  * Survivorship Bias bleibt: Die Daten enthalten nur heute noch gelistete Aktien. Deshalb immer auch gegen
    das gleichgewichtete Universum vergleichen, das denselben Fehler enthält.
"""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

TRADING_DAYS = 252
# Tagesrenditen je Aktie auf diesen Bereich kappen: Größere Sprünge sind in den Yahoo-Daten fast immer
# Datenfehler (z. B. falsch zurückgerechnete Reverse-Splits bei Kleinstwerten)
RET_CLIP = (-0.75, 1.0)


@dataclass
class Panel:
    """Kurse als Tabellen (Zeilen = Handelstage, Spalten = Ticker)."""
    close: pd.DataFrame  # dividenden- und split-bereinigt (Adj Close)
    open: pd.DataFrame  # mit demselben Faktor bereinigt
    raw_close: pd.DataFrame  # nur split-bereinigt, für Preisfilter
    volume: pd.DataFrame

    @property
    def r_cc(self) -> pd.DataFrame:  # Schluss -> Schluss
        return self.close.pct_change(fill_method=None)

    @property
    def r_on(self) -> pd.DataFrame:  # Schluss -> nächste Eröffnung (über Nacht)
        return self.open / self.close.shift(1) - 1

    @property
    def r_id(self) -> pd.DataFrame:  # Eröffnung -> Schluss (Handelstag)
        return self.close / self.open - 1


def load_panel(data_dir: Path, start: str = "2003-01-01") -> Panel:
    path = str(data_dir / "prices" / "1d" / "*.parquet")
    df = duckdb.connect().execute(
        "SELECT ticker, date, open, close, adj_close, volume FROM read_parquet(?, union_by_name=true) WHERE date >= ?",
        [path, start],
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    factor = df["adj_close"] / df["close"]
    df["adj_open"] = np.where(df["open"] > 0, df["open"] * factor, np.nan)
    wide = lambda col: df.pivot(index="date", columns="ticker", values=col).sort_index()
    return Panel(close=wide("adj_close"), open=wide("adj_open"), raw_close=wide("close"), volume=wide("volume"))


def liquid_universe(panel: Panel, top: int, min_price: float = 5.0, min_dollar_volume: float = 5e6,
                    window: int = 60, min_history: int = 252) -> pd.DataFrame:
    """True, wenn eine Aktie an diesem Tag zu den `top` liquidesten gehört – nur mit Daten bis zum Vortag."""
    dollar_volume = (panel.raw_close * panel.volume).rolling(window, min_periods=window // 2).median().shift(1)
    history = panel.close.notna().cumsum()
    eligible = (panel.raw_close.shift(1) >= min_price) & (history >= min_history) & (dollar_volume >= min_dollar_volume)
    rank = dollar_volume.where(eligible).rank(axis=1, ascending=False)
    return rank <= top


@dataclass
class Result:
    name: str
    daily: pd.DataFrame  # Spalten: ret, gross, cost, turnover, long, short
    stats: dict = field(default_factory=dict)


def run(panel: Panel, weights: pd.DataFrame, name: str, cost_bps: float = 5.0) -> Result:
    """Simuliert ein Depot mit echten Beständen.

    `weights`: Zielgewichte je Tag (Anteil am Depotwert, negativ = leerverkauft), festgelegt NACH Börsenschluss.
    Eine Zeile, die komplett leer (NaN) ist, bedeutet „nichts tun“: Die Positionen laufen dann mit den Kursen mit.
    Umgeschichtet wird zur Eröffnung des Folgetags; Kosten fallen nur auf den Abstand zwischen tatsächlich
    gehaltenen (durch Kursbewegung verschobenen) und gewünschten Gewichten an.
    """
    weights = weights.reindex(index=panel.close.index, columns=panel.close.columns)
    trade_rows = weights.notna().any(axis=1).to_numpy()
    target = weights.fillna(0.0).to_numpy()

    r_on, r_id, r_cc = panel.r_on, panel.r_id, panel.r_cc
    # Fehlt der Eröffnungskurs, wird der ganze Tag als Schluss->Schluss gerechnet
    missing_open = (r_on.isna() | r_id.isna()).to_numpy()
    on = np.where(missing_open, r_cc.to_numpy(), r_on.to_numpy())
    on = np.nan_to_num(on, nan=0.0)
    idr = np.where(missing_open, 0.0, np.nan_to_num(r_id.to_numpy(), nan=0.0))
    extreme = (on < RET_CLIP[0]) | (on > RET_CLIP[1]) | (idr < RET_CLIP[0]) | (idr > RET_CLIP[1])
    on, idr = on.clip(*RET_CLIP), idr.clip(*RET_CLIP)

    n_days = len(target)
    h = np.zeros(target.shape[1])  # Gewichte der Bestände zum letzten Schluss
    out = np.zeros((n_days, 6))
    clipped = 0
    started = False
    for t in range(n_days):
        # 1. Über Nacht: Bestände laufen mit
        g_on = float(h @ on[t])
        if g_on > -1:
            h = h * (1 + on[t]) / (1 + g_on)
        # 2. Eröffnung: umschichten, falls am Vortag nach Schluss entschieden wurde
        turnover = 0.0
        if t > 0 and trade_rows[t - 1]:
            turnover = float(np.abs(target[t - 1] - h).sum())
            h = target[t - 1].copy()
            started = started or bool(np.abs(h).sum() > 0)
        cost = turnover * cost_bps / 1e4
        # 3. Handelstag: Bestände laufen mit
        held = h != 0
        clipped += int((extreme[t] & held).sum())
        g_id = float(h @ idr[t])
        if g_id > -1:
            h = h * (1 + idr[t]) / (1 + g_id)
        gross = (1 + g_on) * (1 + g_id) - 1
        ret = (1 + g_on) * (1 - cost) * (1 + g_id) - 1
        out[t] = (ret if started else np.nan, gross, cost, turnover, (h > 0).sum(), (h < 0).sum())

    daily = pd.DataFrame(out, index=panel.close.index, columns=["ret", "gross", "cost", "turnover", "long", "short"])
    daily = daily[daily["ret"].notna()]  # erst ab dem ersten Tag mit Positionen zählen
    return Result(name, daily, {"clipped": clipped})


def buy_and_hold(panel: Panel, ticker: str, name: str) -> Result:
    r = panel.r_cc[ticker].fillna(0.0)
    return Result(name, pd.DataFrame({"ret": r, "gross": r, "cost": 0.0, "turnover": 0.0, "long": 1, "short": 0}))


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Letzter Handelstag jedes Monats."""
    return pd.DatetimeIndex(index.to_series().groupby(index.to_period("M")).max().values)


def only_on(weights: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """Gewichte nur an diesen Tagen setzen; dazwischen leer = Bestände halten."""
    w = weights.copy()
    w.loc[~w.index.isin(dates)] = np.nan
    return w


def equal_weight(universe: pd.DataFrame, rebalance: pd.Index | None = None) -> pd.DataFrame:
    """Alle Aktien des Universums zu gleichen Teilen; umgeschichtet an `rebalance` (Standard: Monatsende)."""
    u = universe.astype(float)
    w = u.div(u.sum(axis=1), axis=0).fillna(0.0)
    return only_on(w, month_ends(w.index) if rebalance is None else rebalance)


def metrics(ret: pd.Series, market: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    equity = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    out = {
        "start": ret.index[0], "end": ret.index[-1],
        "cagr": cagr, "vol": vol,
        "sharpe": ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan,
        "maxdd": (equity / equity.cummax() - 1).min(),
        "total": equity.iloc[-1] - 1,
        "pos_months": (ret.resample("ME").apply(lambda r: (1 + r).prod() - 1) > 0).mean(),
    }
    if market is not None:
        m = market.reindex(ret.index).fillna(0.0)
        beta = np.cov(ret, m)[0, 1] / m.var()
        out["beta"] = beta
        # Alpha: Rendite, die nicht durch das Mitschwingen mit dem Markt erklärt wird (annualisiert)
        out["alpha"] = (ret - beta * m).mean() * TRADING_DAYS
    if turnover is not None:
        # Umschlag pro Jahr: wie oft das Depot im Jahr komplett umgeschichtet wird (Kauf + Verkauf = 2)
        out["turnover_year"] = turnover.reindex(ret.index).mean() * TRADING_DAYS / 2
    return out


def evaluate(result: Result, market: pd.Series, split: str) -> Result:
    """Kennzahlen gesamt, vor `split` (In-Sample) und ab `split` (Out-of-Sample)."""
    d = result.daily
    result.stats = {
        "clipped": result.stats.get("clipped", 0),
        "all": metrics(d["ret"], market, d["turnover"]),
        "is": metrics(d.loc[d.index < split, "ret"], market, d["turnover"]),
        "oos": metrics(d.loc[d.index >= split, "ret"], market, d["turnover"]),
        "years": d["ret"].groupby(d.index.year).apply(lambda r: (1 + r).prod() - 1),
    }
    return result
