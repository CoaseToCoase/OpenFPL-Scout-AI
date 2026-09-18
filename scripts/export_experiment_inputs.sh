#!/usr/bin/env bash
# One-off export of experiment inputs from the OctoFPL-vAI DB on MMMS.
# Read-only. Run from the repo root:  bash scripts/export_experiment_inputs.sh
set -euo pipefail

HOST=monaco-mac-mini-server
DB=~/OctoFPL-vAI/db/octofpl.db
OUT=data/official
mkdir -p "$OUT"

# Point-in-time ep_next. Provenance matters: combined.csv's own
# expected_points is vaastav's xP, scraped AFTER each gameweek, so using it
# unshifted would leak. fpl_ep_next_history is rebuilt from pre-deadline
# snapshots and its loader refuses any snapshot at or after its deadline.
ssh "$HOST" "sqlite3 -header -csv $DB \"
  SELECT season, element_id, gw, ep_next, player_code
  FROM fpl_ep_next_history
  WHERE season IN ('2023_24','2024_25','2025_26')
  ORDER BY season, gw, element_id\"" > "$OUT/ep_next_pit.csv"

# Prior-season aggregates, for season N-1 of each season in combined.csv.
# gw_player_stats_historical is code-keyed and has no 'starts' column, so
# appearances (minutes > 0) and total minutes carry that signal.
ssh "$HOST" "sqlite3 -header -csv $DB \"
  SELECT season, player_code,
         SUM(minutes > 0) AS appearances,
         SUM(minutes)     AS minutes,
         SUM(points)      AS points
  FROM gw_player_stats_historical
  WHERE season IN ('2022_23','2023_24','2024_25')
  GROUP BY season, player_code
  ORDER BY season, player_code\"" > "$OUT/prior_season_stats.csv"

wc -l "$OUT/ep_next_pit.csv" "$OUT/prior_season_stats.csv"
