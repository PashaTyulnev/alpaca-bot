# aktien-bot-v1 – Datenimport

Lädt historische Kurse kostenlos über [`yfinance`](https://github.com/ranaroussi/yfinance) (Yahoo Finance) für alle Nasdaq-Stammaktien und speichert sie als Parquet-Dateien, die sich mit pandas oder DuckDB (SQL) abfragen lassen.

## Installation (Windows, PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Daten laden

```powershell
.venv\Scripts\python download_data.py
```

Zum Ausprobieren mit wenigen Tickern:

```powershell
.venv\Scripts\python download_data.py --limit 20
```

Die Einstellungen stehen in `config.toml`:
- `universe`: Universum, Standard ist `nasdaq-common`
- `tickers`: zusätzliche Ticker, z. B. SPY
- `intervals`: Intervalle
- `pause_seconds`: Pause zwischen den Anfragen
- `skip_if_younger_than_hours`: Überspringen, wenn eine Datei jünger ist

Der erste vollständige Lauf dauert etwa **1,5–2 Stunden** (ca. 3.400 Ticker × 2 Intervalle) und belegt rund **0,8 GB**. Wird er abgebrochen, startest du ihn einfach neu. Ticker, die in den letzten 12 Stunden geladen wurden, werden übersprungen.

Fehler bei einzelnen Tickern brechen den Lauf nicht ab. Sie stehen am Ende in der Zusammenfassung und in `data/import_log.csv`. Eine vorhandene Datei wird bei einem Fehler nie überschrieben. Exit-Code 1 bedeutet, dass mindestens ein Ticker fehlgeschlagen ist. Drosselt Yahoo dauerhaft, wartet das Skript bis zu ca. 14 Minuten und bricht dann sauber ab.

## Ablage

```
data/
  universe/nasdaq-common.csv     Ticker-Liste mit Namen (Stand des letzten Laufs)
  prices/1d/<TICKER>.parquet     Tageskerzen, komplette Historie
  prices/1h/<TICKER>.parquet     Stundenkerzen
  import_log.csv                 Ergebnis jedes Tickers je Lauf
  overview.html                  Übersichtsseite
```

Spalten: `ticker, date|timestamp, open, high, low, close, adj_close, volume, dividends, stock_splits` (bei Fonds zusätzlich `capital_gains`).

- `close` ist **split-bereinigt**, aber nicht dividendenbereinigt. `adj_close` ist split- **und** dividendenbereinigt.
- `dividends` ist der Betrag je Aktie am Ex-Tag, `stock_splits` der Faktor (z. B. `4.0`). Sonst steht dort `0`.
- 1h-Zeitstempel sind in New Yorker Börsenzeit gespeichert. Eine Kerze beginnt zum angegebenen Zeitpunkt (09:30, 10:30, …).

`data/` ist über `.gitignore` von Git ausgeschlossen.

## Abfragen mit DuckDB

```python
import duckdb
duckdb.sql("""
    SELECT ticker, date, adj_close
    FROM 'data/prices/1d/*.parquet'
    WHERE ticker IN ('AAPL', 'MSFT') AND date >= '2020-01-01'
""").df()
```

## Übersicht der vorhandenen Daten

```powershell
.venv\Scripts\python overview.py
```

Erzeugt `data/overview.html` und öffnet die Seite im Browser. Sie zeigt:
- Kennzahlen pro Intervall
- pro Ticker den Zeitraum (von/bis) und die Zahl der Kerzen je Intervall
- einen Zeitbalken für die Tagesdaten
- Probleme: fehlende Daten mit Fehlermeldung aus dem Import, veraltete Daten, unlesbare Dateien

Mit der Suche und dem Filter „nur Probleme“ grenzt du die Liste ein. Die Seite ist eine Momentaufnahme, also nach jedem Import neu erzeugen.

## Korrelationen

```powershell
.venv\Scripts\python correlations.py
```

Erzeugt `data/correlations.html`. Die Seite ist auch aus der Übersicht verlinkt. Sie enthält:
- eine Heatmap der Rendite-Korrelationen der 500 liquidesten Aktien, nach Ähnlichkeit geclustert
- Zeitfenster von 1 Woche, 2 Wochen und 1 Monat (aus Stundenrenditen) sowie 3 Monaten und 1 Jahr (aus Tagesrenditen)
- zwei Varianten: roh und marktbereinigt (ohne den SPY-Anteil)
- Listen der stärksten Gleich- und Gegenläufer
- eine Partnerliste pro Ticker

Mit `--top 1000` nimmst du mehr Aktien auf, die Seite wird dann größer.

## Intraday-Daten von Alpaca

Zugangsdaten stehen in `.env` (`ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`). Die Datei ist nicht in Git.

```powershell
.venv\Scripts\python alpaca_download.py
```

Lädt 5-Minuten-Kerzen des Gesamtmarkts (SIP) für SPY und die 500 liquidesten Aktien ab 2016 nach `data/alpaca/5Min/`.
- Jede Kerze hat eine Spalte `session`: `pre`, `regular` oder `post`, laut Börsenkalender inklusive verkürzter Handelstage.
- Kurse sind split-bereinigt.
- Der kostenlose Zugang liefert ca. 2.600 Kerzen pro Anfrage und erlaubt 200 Anfragen pro Minute. Der erste Lauf dauert daher ca. 8–9 Stunden.
- Weitere Läufe laden nur neue Kerzen.

Optionen:
- `--timeframe 1Min` für feinere Kerzen
- `--top`, um die Zahl der Aktien zu ändern
- `--tickers AAPL,MSFT` für bestimmte Ticker
- `--start 2020-01-01` für einen anderen Startzeitpunkt

## Eröffnungs-Analysen

```powershell
.venv\Scripts\python analysis_open.py
```

Erzeugt `data/analysis.html` mit drei Ereignis-Studien:
1. Eröffnungslücke (Gap) und Tagesverlauf
2. Lage um 10:00 zum Vortageshoch und -tief
3. Opening Range Breakout (Ausbruch aus der Spanne der ersten 30 Minuten)

Gemessen wird 2016–2022. Die daraus abgeleitete Richtung wird ab 2023 unverändert nach Kosten geprüft (Standard 5 bps, ändern mit `--cost-bps`). Die Tageskennzahlen je Aktie landen zusätzlich in `data/alpaca/day_stats_5Min.parquet` für spätere Backtests.

## Backtests auf Tagesbasis

```powershell
.venv\Scripts\python strategies_daily.py
```

`backtest.py` ist das Backtest-Gerüst:
- Zielgewichte je Tag, Signal nach Börsenschluss, Umschichtung zur Eröffnung am Folgetag
- Kosten je gehandeltem Dollar
- Universum aus der damaligen Liquidität (mindestens 5 Mio. $ Tagesumsatz)
- Kennzahlen: Rendite, Sharpe, maximaler Verlust, Beta, Alpha, Umschlag

`strategies_daily.py` testet drei Strategien, jeweils als Long und als Long-Short:
- Momentum 12-1
- kurzfristige Umkehr (5 Tage)
- Drift nach Kurslücke

Getestet wird bis 2015 und ab 2016, verglichen mit SPY und dem gleichgewichteten Universum. Die Ergebnisse stehen in `data/backtests.html`.

## Marktbreite

```powershell
.venv\Scripts\python breadth.py
```

Prüft eine vorab festgelegte Regel: Wenn ≥ 50 % der Aktien unter bzw. über ihre 30-Minuten-Spanne fallen, wird SPY bis Schluss gehandelt. Die Ergebnisse stehen in `data/breadth.html`.

## Fundamentaldaten (SEC EDGAR)

Daten herunterladen (ca. 1,4 GB). Die SEC verlangt einen Kontakt im User-Agent, siehe `SEC_USER_AGENT` in `.env`:

```powershell
curl.exe -A "aktien-bot-v1 deine@mail" -o data\sec\companyfacts.zip https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
curl.exe -A "aktien-bot-v1 deine@mail" -o data\sec\company_tickers.json https://www.sec.gov/files/company_tickers.json
```

Danach auswerten:

```powershell
.venv\Scripts\python sec_fundamentals.py
.venv\Scripts\python fundamentals_test.py
```

- `sec_fundamentals.py` extrahiert zeitpunktgenaue Jahreszahlen. Jeder Wert gilt ab seiner ersten Veröffentlichung, spätere Korrekturen werden ignoriert.
- `fundamentals_test.py` vergleicht elf Kennzahlen mit der Rendite der folgenden 1, 3 und 12 Monate (Rangkorrelation) und testet sie als Long-Short-Portfolio. Die Richtungen stehen vorab aus der Forschung fest. Getestet wird bis 2017 und ab 2018. Die Ergebnisse stehen in `data/fundamentals.html`.

## Long-only mit Qualitäts- und Bewertungsfilter

```powershell
.venv\Scripts\python long_only.py
```

Voraussetzung: Vorher muss `fundamentals_test.py` gelaufen sein, das erzeugt `data/sec/factors_monthly.parquet`.

Hauptvariante und Erfolgskriterien sind im Kopf von `long_only.py` **vor der ersten Berechnung** festgelegt.

Die Hauptvariante:
- Kauft monatlich die besten 20 % nach Gewinnrendite, Free-Cashflow-Rendite, Eigenkapitalrendite, Bruttogewinn/Vermögen und wenig neuen Aktien.
- Gewichtet gleich und rechnet 10 bps Kosten.

Geprüft wird:
- gegen SPY, das gleichgewichtete und das kapitalgewichtete Universum
- gegen 200 Zufallsdepots
- mit 7 Varianten zur Robustheit

Die Ergebnisse stehen in `data/long_only.html`.

## Paper-Trading auf dem Server (Linux)

`paper_trade.py` handelt die Q+V-Strategie (Hauptvariante aus `long_only.py`) auf dem Alpaca-**Paper**-Konto. Die nötigen Daten holt es selbst: Kurse von Alpaca, Fundamentaldaten von der SEC. Einen `data/`-Ordner aus der lokalen Analyse braucht es nicht.

Einrichten:

```bash
git clone https://github.com/PashaTyulnev/alpaca-bot.git && cd alpaca-bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env      # Alpaca-Keys und SEC_USER_AGENT eintragen
chmod +x run_paper.sh
.venv/bin/python paper_trade.py plan   # Probelauf: Zielauswahl berechnen, handelt nicht
```

Cronjob einrichten (`crontab -e`), Pfad anpassen:

```
*/15 * * * 1-5 /home/DEINUSER/alpaca-bot/run_paper.sh tick
```

Was `tick` tut, gesteuert über die Börsenuhr von Alpaca (die Server-Zeitzone ist egal):
- **Erster Handelstag des Monats:** vorbörslich die Zielauswahl berechnen und 15 Minuten nach Börsenstart umschichten. Beim allerersten Start passiert das schon am nächsten Handelstag.
- **Nach Börsenschluss:** Wert der Strategie und SPY-Schlusskurs in `data/live/performance.csv` schreiben.
- **Sonst:** nichts.

Ansehen:

```bash
.venv/bin/python paper_trade.py status   # Positionen und Vergleich mit SPY
tail -f data/live/paper.log               # Protokoll
```

Sicherheit:
- Das Skript startet nur, wenn `ALPACA_TRADING_URL` auf `paper-api` zeigt.
- Es verwaltet nur Stückzahlen, die es selbst gekauft hat, und nur das Bargeld, das beim ersten Start frei war. Die Buchhaltung steht in `data/live/state.json`. Andere Positionen im Konto bleiben unberührt.

Hinweise:
- Bei kleinem Konto (z. B. 10.000 $) entfallen auf jede der rund 95 Aktien nur ~100 $. Nicht fraktionierbare, teure Aktien werden dann übersprungen, das steht im Log. Für einen aussagekräftigen Test das Paper-Konto im Alpaca-Dashboard auf z. B. 100.000 $ zurücksetzen.
- Dividenden schreibt Alpaca dem Konto gut, die Buchhaltung der Strategie erfasst sie aber nicht. Der Vergleich in `status` nutzt deshalb ebenfalls SPY ohne Dividenden.

## Einschränkungen der Datenquelle

- **Keine offizielle API.** Yahoo Finance hat keine offizielle Schnittstelle, `yfinance` liest inoffizielle Endpunkte. Das kann ohne Vorwarnung kaputtgehen oder gedrosselt werden. Die Nutzung ist laut Yahoo nur für private Zwecke gedacht.
- **Begrenzte Intraday-Historie.** 1h-Daten gibt es nur für die letzten 730 Tage. Das Skript führt sie bei jedem Lauf mit den vorhandenen zusammen. Wer regelmäßig lädt, baut sich so eine längere Stundenhistorie auf. Nach einem späteren Split passen ältere, bereits gespeicherte 1h-Kerzen aber nicht mehr zur neuen Bereinigung.
- **Survivorship Bias.** Das Universum enthält nur die **heute** gelisteten Aktien. Delistete und insolvente Firmen fehlen, was Backtests zu optimistisch macht.
- **SPACs zählen mit.** Die Nasdaq-Liste führt SPACs („Acquisition Corp“) als Stammaktien, deshalb sind sie enthalten.
- **Rückwirkende Bereinigung.** Nach neuen Splits oder Dividenden ändern sich ältere Tageswerte. Tagesdaten werden deshalb bei jedem Lauf komplett ersetzt.
- **Unvollständige Kerze.** Läuft der Import während der Handelszeit, ist die letzte Kerze unvollständig.
- **Keine Qualitätsgarantie.** Gelegentlich gibt es fehlende Tage oder Ausreißer.
