"""Zeitpunktgenaue Fundamentaldaten aus SEC EDGAR (companyfacts.zip) für die Ticker aus data/prices/1d.

Ablauf:
  1. Relevante XBRL-Kennzahlen aus data/sec/companyfacts.zip lesen -> data/sec/facts.parquet
  2. Geschäftsjahres-Tabelle bauen: je Firma und Geschäftsjahr die ZUERST veröffentlichten Werte samt
     Veröffentlichungsdatum (spätere Korrekturen werden ignoriert, weil sie damals noch unbekannt waren)
     -> data/sec/annual.parquet
  3. Aktienanzahl je Veröffentlichung -> data/sec/shares.parquet

Download der Rohdaten (ca. 1,4 GB, Kontakt laut SEC-Vorgabe aus .env: SEC_USER_AGENT):
  https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
  https://www.sec.gov/files/company_tickers.json

Start:  python sec_fundamentals.py
"""

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from download_data import PROJECT_DIR, load_config, resolve_data_dir

# Kennzahl -> mögliche XBRL-Namen (erste vorhandene gewinnt je Firma und Periode)
FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
INSTANT = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
}
SHARES = ("dei", "EntityCommonStockSharesOutstanding")
ANNUAL_FORMS = {"10-K", "10-K405", "10-KT", "10-K/A"}


def ticker_to_cik(data_dir: Path) -> dict[str, int]:
    raw = json.loads((data_dir / "sec" / "company_tickers.json").read_text(encoding="utf-8"))
    sec = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}
    ours = {p.stem for p in (data_dir / "prices" / "1d").glob("*.parquet")}
    return {t: sec[t] for t in ours if t in sec}


def extract_facts(data_dir: Path, mapping: dict[str, int]) -> pd.DataFrame:
    wanted = {**FLOW, **INSTANT}
    cik_to_tickers: dict[int, list[str]] = {}
    for t, c in mapping.items():
        cik_to_tickers.setdefault(c, []).append(t)
    rows = []
    with zipfile.ZipFile(data_dir / "sec" / "companyfacts.zip") as z:
        names = {int(n[3:13]): n for n in z.namelist() if n.startswith("CIK")}
        for i, (cik, tickers) in enumerate(cik_to_tickers.items(), 1):
            if cik not in names:
                continue
            facts = json.loads(z.read(names[cik])).get("facts", {})
            gaap = facts.get("us-gaap", {})
            for key, concepts in wanted.items():
                for prio, concept in enumerate(concepts):
                    for unit, items in gaap.get(concept, {}).get("units", {}).items():
                        if unit != "USD":
                            continue
                        for it in items:
                            rows.append((cik, key, prio, it.get("start"), it["end"], it["val"], it["filed"], it.get("form")))
            for it in facts.get(SHARES[0], {}).get(SHARES[1], {}).get("units", {}).get("shares", []):
                rows.append((cik, "shares", 0, None, it["end"], it["val"], it["filed"], it.get("form")))
            if i % 500 == 0:
                print(f"  {i}/{len(cik_to_tickers)} Firmen gelesen")
    df = pd.DataFrame(rows, columns=["cik", "key", "prio", "start", "end", "val", "filed", "form"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    return df


def build_annual(facts: pd.DataFrame) -> pd.DataFrame:
    """Je Firma und Geschäftsjahresende die zuerst veröffentlichten Jahreswerte."""
    f = facts[facts["form"].isin(ANNUAL_FORMS)].copy()
    dur = (f["end"] - f["start"]).dt.days
    flows = f[f["key"].isin(FLOW) & dur.between(350, 380)]
    inst = f[f["key"].isin(INSTANT) & f["start"].isna()]
    parts = []
    for df in (flows, inst):
        # Erste Veröffentlichung je (Firma, Kennzahl, Periodenende); bei mehreren XBRL-Namen der bevorzugte
        df = df.sort_values(["filed", "prio"]).drop_duplicates(["cik", "key", "end"], keep="first")
        parts.append(df)
    long = pd.concat(parts)
    # Geschäftsjahresenden = Enden der Jahres-Flussgrößen; Bestandsgrößen am selben Stichtag
    fy_ends = flows[["cik", "end"]].drop_duplicates()
    long = long.merge(fy_ends, on=["cik", "end"])
    annual = long.pivot_table(index=["cik", "end"], columns="key", values="val", aggfunc="first")
    # Verfügbar ab dem Tag, an dem ALLE genutzten Werte veröffentlicht waren (konservativ)
    annual["available"] = long.groupby(["cik", "end"])["filed"].max()
    if "gross_profit" in annual and "cogs" in annual:
        annual["gross_profit"] = annual["gross_profit"].fillna(annual["revenue"] - annual["cogs"])
    return annual.reset_index().sort_values(["cik", "end"])


def build_shares(facts: pd.DataFrame) -> pd.DataFrame:
    s = facts[(facts["key"] == "shares") & (facts["val"] > 0)]
    return (s.sort_values("filed").drop_duplicates(["cik", "end"], keep="first")[["cik", "end", "val", "filed"]]
            .rename(columns={"end": "shares_date", "val": "shares"}))


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC-Fundamentaldaten zeitpunktgenau aufbereiten.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    args = parser.parse_args()
    data_dir = resolve_data_dir(load_config(args.config))
    if not (data_dir / "sec" / "companyfacts.zip").exists():
        raise SystemExit("data/sec/companyfacts.zip fehlt – siehe Kopf dieser Datei für den Download.")

    mapping = ticker_to_cik(data_dir)
    print(f"{len(mapping)} unserer Ticker haben eine SEC-Nummer (CIK)")
    facts = extract_facts(data_dir, mapping)
    facts.to_parquet(data_dir / "sec" / "facts.parquet", index=False)
    print(f"{len(facts):,} Einzelwerte gelesen".replace(",", "."))

    annual = build_annual(facts)
    annual.to_parquet(data_dir / "sec" / "annual.parquet", index=False)
    shares = build_shares(facts)
    shares.to_parquet(data_dir / "sec" / "shares.parquet", index=False)
    pd.Series(mapping, name="cik").rename_axis("ticker").reset_index().to_parquet(data_dir / "sec" / "tickers.parquet", index=False)

    print(f"Geschäftsjahre: {len(annual):,} von {annual['cik'].nunique():,} Firmen, "
          f"{annual['end'].min():%Y} bis {annual['end'].max():%Y}".replace(",", "."))
    print("Abdeckung je Kennzahl:", {k: f"{annual[k].notna().mean():.0%}" for k in [*FLOW, *INSTANT] if k in annual})
    lag = (annual["available"] - annual["end"]).dt.days
    print(f"Veröffentlichung nach Geschäftsjahresende: Median {lag.median():.0f} Tage, 90 % innerhalb {lag.quantile(.9):.0f} Tagen")


if __name__ == "__main__":
    main()
