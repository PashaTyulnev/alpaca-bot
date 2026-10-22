#!/bin/sh
# Startet paper_trade.py für Cron: im Projektordner, ohne Überlappung (flock), Ausgabe ins Log.
# Crontab-Beispiel (alle 15 Minuten, Mo–Fr):
#   */15 * * * 1-5 /pfad/zu/alpaca-bot/run_paper.sh tick
cd "$(dirname "$0")" || exit 1
mkdir -p data/live
exec flock -n data/live/.lock .venv/bin/python paper_trade.py "${1:-tick}" >> data/live/paper.log 2>&1
