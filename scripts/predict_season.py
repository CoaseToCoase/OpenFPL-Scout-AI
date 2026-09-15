"""Project every remaining gameweek, not just the next one.

Why this exists: OctoFPL-vAI compares rest-of-season projections from FFS,
PLFantasyTool and FPL Form, and our own model could not take part because
predict_gameweek.py produces exactly one gameweek.

How it works, and the honest limits of it:

  * Form features are FROZEN at the last five matches played before the first
    projected gameweek. They are not re-estimated as the season runs, because
    the inputs for that do not exist yet — so a projection for GW30 is "this
    player, in this form, against that opponent".
  * What DOES vary per gameweek is the fixture: opponent_team_name, was_home
    and the gameweek number, taken from the official fixture list. That is the
    same lever the commercial sites pull.
  * A double gameweek sums its fixtures; a blank scores zero.
  * Nothing models injuries, suspensions, transfers, role changes or form
    decay. Treat the far end of the horizon as a fixture-weighted view of
    current form, NOT a forecast of what the player will be in April.

**Fixture attachment is also a correctness fix, not only a new feature.**
`prepare_recent_player_features` fills opponent_team_name / was_home from the
player's most recent PLAYED match, so the single-gameweek script has been
predicting each player against the team they last faced rather than the one
they are about to face. Training rows carry each match's own opponent, so the
upcoming fixture is what inference should use. `--compare-gw` prints the
difference that makes, so the effect is measured rather than asserted.

Usage:
    uv run --group train python3 scripts/predict_season.py [--from-gameweek N]
    uv run --group train python3 scripts/predict_season.py --compare-gw 4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_gameweek import MODEL_NAMES, MODELS_DIR, load_ensemble  # noqa: E402
from src.features import prepare_recent_player_features  # noqa: E402
from src.fixture_projection import predict_frame, project, team_fixtures  # noqa: E402
from src.official_fpl import OfficialFPLClient  # noqa: E402

LAST_GAMEWEEK = 38


def form_gameweek_status(bootstrap: dict, gameweek: int) -> dict:
    """Was the gameweek the form is frozen at FINAL when this run was made?

    Recorded at generation time because it cannot be reconstructed later:
    FPL flips `data_checked` once bonus is confirmed, and a run made before
    that used provisional points. Checking the gameweek's status next week
    would call this run final when it was not (2026-09-15: the from-GW5 run
    was made while all ten GW4 fixtures were still finished_provisional).
    A gameweek that is not in the events list (form frozen at GW0) has no
    data to be provisional about, so it counts as final.
    """
    event = next((e for e in bootstrap.get("events", []) if e.get("id") == gameweek), None)
    if event is None:
        return {"finished": True, "data_checked": True}
    return {"finished": bool(event.get("finished")),
            "data_checked": bool(event.get("data_checked"))}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-gameweek", type=int,
                        help="First gameweek to project (default: the next one)")
    parser.add_argument("--to-gameweek", type=int, default=LAST_GAMEWEEK)
    parser.add_argument("--history-window", type=int, default=5,
                        help="Rolling-history window (must match training: 5)")
    parser.add_argument("--compare-gw", type=int,
                        help="Report how the fixture fix changes this gameweek, then exit")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    client = OfficialFPLClient()
    bootstrap = client.bootstrap()
    first_gw = int(args.compare_gw or args.from_gameweek or client.next_gameweek())
    season_start = int(bootstrap["events"][0]["deadline_time"][:4])
    season = f"{season_start}-{str(season_start + 1)[-2:]}"

    history = client.player_history(first_gw, selectable_only=False)
    base = prepare_recent_player_features(history, gameweek=first_gw,
                                          history_window=args.history_window)
    models = load_ensemble()
    fixtures = team_fixtures(client)

    if args.compare_gw:
        before = predict_frame(models, base)
        after = project(models, base, fixtures, [args.compare_gw])
        rows = []
        for i, r in base.reset_index(drop=True).iterrows():
            new = after.get(int(r["id"]), {}).get(args.compare_gw)
            if new is not None:
                rows.append((r["web_name"], float(before[i]), new, new - float(before[i])))
        if not rows:
            print("no overlapping players — nothing to compare")
            return 1
        deltas = [abs(d) for *_, d in rows]
        rows.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"GW{args.compare_gw}: fixture fix changes {len(rows)} players, "
              f"mean |delta| {np.mean(deltas):.3f}, max {max(deltas):.3f}")
        print("biggest moves (last-opponent -> next-opponent):")
        for name, b, a, d in rows[:10]:
            print(f"  {name:<16}{b:>6.2f} -> {a:>6.2f}  ({d:+.2f})")
        return 0

    gws = [g for g in range(first_gw, args.to_gameweek + 1)]
    projected = project(models, base, fixtures, gws)
    trained_at = None
    metadata_path = MODELS_DIR / "training_metadata.json"
    if metadata_path.exists():
        trained_at = json.loads(metadata_path.read_text()).get("trained_at_utc")

    predictions = []
    for _, r in base.reset_index(drop=True).iterrows():
        eid = int(r["id"])
        by_gw = projected.get(eid, {})
        predictions.append({
            "element_id": eid,
            "web_name": r["web_name"],
            "team_name": r["team_name"],
            "element_type": int(r["element_type"]),
            "points_by_gameweek": {str(g): round(by_gw.get(g, 0.0), 3) for g in gws},
            "total_points": round(sum(by_gw.values()), 3),
        })
    predictions.sort(key=lambda p: -p["total_points"])

    output = {
        "source": "openfpl-scout-fork-season",
        "season": season,
        "from_gameweek": first_gw,
        "to_gameweek": args.to_gameweek,
        "model_trained_at_utc": trained_at,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "form_frozen_at_gameweek": first_gw - 1,
        "form_gameweek_status": form_gameweek_status(bootstrap, first_gw - 1),
        "player_count": len(predictions),
        "predictions": predictions,
    }
    out_dir = REPO_ROOT / "data" / "predictions" / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"season_from_gw{first_gw}.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(predictions)} season projections (GW{first_gw}-{args.to_gameweek}) -> {out_path}")
    if not output["form_gameweek_status"]["data_checked"]:
        print(f"WARNING: GW{first_gw - 1} is not data_checked yet -- this run uses PROVISIONAL "
              f"points and movers reports will not compare it. Re-run once FPL confirms bonus.")
    print(f"Upload with: gsutil cp {out_path} "
          f"gs://octosuitedatahub/openfpl-scout/{season}/season_from_gw{first_gw}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
