"""End-to-end: load F1DB -> features -> backtest -> final models -> dashboard/data.json.

    python -m f1dash.build              # reuse cached backtests if present
    python -m f1dash.build --backtest   # recompute the walk-forward backtests (~6 min)
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build as build_features
from .load import load
from .model import RaceModel, calibration, score, walk_forward
from .strategy import circuit_profile, season_poles

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

TEAM_COLORS = {
    "mercedes": "#00D2BE", "ferrari": "#E8002D", "red-bull": "#3671C6", "mclaren": "#FF8000",
    "aston-martin": "#229971", "alpine": "#0093CC", "williams": "#64C4FF", "racing-bulls": "#6692FF",
    "haas": "#B6BABD", "audi": "#9C1B2F", "cadillac": "#C9A227",
}
FEATURE_LABELS = {
    "d_finish_ewm": "Recent finishing positions", "d_qpos_ewm": "Recent qualifying positions",
    "d_qgap_ewm": "Recent qualifying gap to pole", "d_pts_last5": "Points per race, last 5",
    "d_win_rate10": "Win rate, last 10", "d_podium_rate10": "Podium rate, last 10", "d_pole_rate10": "Pole rate, last 10",
    "d_dnf_rate10": "Retirement rate, last 10", "d_tm_quali_delta": "Qualifying vs team-mate",
    "d_circ_avg_finish": "Past results at this circuit", "d_circ_wins": "Past wins at this circuit",
    "d_circ_starts": "Starts at this circuit", "d_starts": "Career starts",
    "t_gap_last5": "Team qualifying pace, last 5", "t_qpos_last5": "Team best grid, last 5",
    "t_pts_last5": "Team points, last 5", "t_best_finish_last5": "Team best finish, last 5",
    "c_mean_abs_gain": "Circuit overtaking", "c_pole_win_rate": "Circuit pole-to-win rate", "round_frac": "Point in season",
    "d_finish_ewm_rank": "Recent finishes (rank in field)", "d_qpos_ewm_rank": "Recent qualifying (rank in field)",
    "d_qgap_ewm_rank": "Qualifying gap (rank in field)", "d_pts_last5_rank": "Recent points (rank in field)",
    "d_season_pts_rank": "Championship position", "t_gap_last5_rank": "Team pace (rank in field)",
    "t_pts_last5_rank": "Team points (rank in field)", "t_season_pts_rank": "Constructors' position",
    "t_gap_vs_best": "Team pace gap to fastest car", "d_season_pts_share": "Season points vs leader",
    "d_pts_last5_vs_best": "Recent points vs best driver",
}


POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def _clinch_scenarios(ds: pd.DataFrame, form: dict, max_left: int, races_left: int) -> list[dict]:
    """For each rival still in contention: if the rival wins the next race, the worst finish that
    still seals the title for the leader there. Ties are broken on number of wins (FIA rule)."""
    leader = ds.iloc[0]
    lw = form.get(leader.driverId, {}).get("wins", 0)
    after = max_left - 25  # points available after the next race (assumes no sprint next)
    out = []
    for r in ds.iloc[1:].itertuples():
        gap = float(leader.points - r.points)
        if gap > max_left:
            continue
        rw = form.get(r.driverId, {}).get("wins", 0)
        worst = None
        for pos in range(21, 0, -1):  # worst finish first
            x = POINTS[pos - 1] if pos <= 10 else 0
            lead = gap - 25 + x
            # rival then wins every remaining race; leader scores nothing more
            if lead > after or (lead == after and lw > rw + 1 + (races_left - 1)):
                worst = pos
                break
        out.append({"rival": r.driverId, "gap": gap, "worst_finish_if_rival_wins": worst})
    return out


def _backtests(train: pd.DataFrame, recompute: bool) -> dict[str, pd.DataFrame]:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for t in ("win", "pole", "podium"):
        f = CACHE / f"backtest_{t}.pkl"
        if recompute or not f.exists():
            print(f"backtesting {t} ...", flush=True)
            walk_forward(train, t, 2022).to_pickle(f)
        out[t] = pd.read_pickle(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="recompute walk-forward backtests")
    args = ap.parse_args(argv)

    d = load(1990)
    drivers, teams, gps, circuits = d["drivers"], d["constructors"], d["grands_prix"], d["circuits"]
    print("building features ...", flush=True)
    train, upcoming = build_features(d)
    season = int(d["races"][d["races"].completed].season.max())
    bts = _backtests(train, args.backtest)

    # ---- backtest summary ----
    evaluation = {}
    for t, bt in bts.items():
        if t == "podium":
            evaluation[t] = {
                "brier": float(((bt.p - bt.podium) ** 2).mean()),
                "brier_base": float(((bt.p_base.clip(upper=1) - bt.podium) ** 2).mean()),
                "calibration": calibration(bt, "p", "podium"),
            }
            continue
        sc = score(bt, t)
        by_season = []
        for s, g in bt.groupby("season"):
            x = score(g, t, n_boot=1)
            by_season.append({"season": int(s), "races": int(g.race_id.nunique()), "top1": x["p"]["top1"][0],
                              "top1_base": x["p_base"]["top1"][0], "ll": x["p"]["ll"][0], "ll_base": x["p_base"]["ll"][0]})
        evaluation[t] = {"overall": {k: sc[k] for k in ("p", "p_lr", "p_gbm", "p_base")},
                         "ll_gain": sc["ll_improvement_vs_base"], "n_races": sc["n_races"],
                         "uniform_ll": sc["uniform_ll"], "by_season": by_season,
                         "calibration": calibration(bt, "p", t)}

    # ---- this season's track record (walk-forward picks, made before each race) ----
    races = d["races"].set_index("race_id")
    record = []
    bw, bp = bts["win"], bts["pole"]
    for rid, g in bw[bw.season == season].groupby("race_id"):
        fav = g.loc[g.p.idxmax()]
        win = g.loc[g.win == 1].iloc[0]
        gp_ = bp[bp.race_id == rid]
        pfav = gp_.loc[gp_.p.idxmax()] if len(gp_) else None
        pact = gp_[gp_.pole == 1]
        record.append({
            "round": int(races.loc[rid, "round"]), "gp": races.loc[rid, "grand_prix_id"],
            "pick": fav.driver_id, "pick_p": float(fav.p), "winner": win.driver_id, "winner_p": float(win.p),
            "winner_rank": int((g.p > win.p).sum() + 1), "hit": bool(fav.win == 1),
            "pole_pick": pfav.driver_id if pfav is not None else None,
            "pole": pact.driver_id.iloc[0] if len(pact) else None,
            "pole_hit": bool(pfav is not None and pfav.pole == 1),
        })
    record.sort(key=lambda r: r["round"])

    # ---- final models on every completed race ----
    print("training final models ...", flush=True)
    models = {t: RaceModel(t).fit(train, season + (0 if upcoming.season.max() == season else 1))
              for t in ("win", "pole", "podium")}
    preds = {t: m.predict(upcoming) for t, m in models.items()}
    up = upcoming[["race_id", "season", "round", "driver_id", "constructor_id", "d_pts_last5", "d_season_pts",
                   "d_circ_starts", "d_circ_avg_finish", "d_circ_wins", "t_gap_last5", "d_qpos_ewm", "d_finish_ewm"]].copy()
    for t in preds:
        up[f"p_{t}"] = preds[t].p.to_numpy()
    imp = models["win"].importance()
    importance = [{"feature": FEATURE_LABELS.get(k, k), "share": float(v)} for k, v in imp.head(10).items()]

    # ---- per-driver recent form (this season) ----
    res = d["results"].merge(d["races"][["race_id", "season", "round"]], on="race_id")
    cur = res[res.season == season].sort_values("round")
    q = d["quali"].merge(d["races"][["race_id", "season", "round"]], on="race_id")
    q = q[q.season == season]
    form = {}
    for drv, g in cur.groupby("driver_id"):
        qq = q[q.driver_id == drv].set_index("round").q_pos
        form[drv] = {
            "results": [{"round": int(r.round), "pos": (int(r.position) if r.classified else r.status),
                         "grid": (int(r.grid) if pd.notna(r.grid) else None), "q": (int(qq.get(r.round)) if pd.notna(qq.get(r.round, np.nan)) else None),
                         "pts": float(r.points)} for r in g.itertuples()],
            "wins": int((g.position == 1).sum()), "podiums": int((g.position <= 3).sum()),
            "poles": int((q[q.driver_id == drv].q_pos == 1).sum()), "dnfs": int((~g.classified).sum()),
            "team": g.constructor_id.iloc[-1],
        }

    # ---- upcoming races ----
    upcoming_out = []
    for rid, g in up.groupby("race_id", sort=False):
        r = races.loc[rid]
        g = g.sort_values("p_win", ascending=False)
        upcoming_out.append({
            "race_id": int(rid), "round": int(r["round"]), "gp": r.grand_prix_id,
            "gp_name": gps.get(r.grand_prix_id, {}).get("full_name") or r.official_name,
            "official_name": r.official_name, "date": str(r.date.date()), "quali_date": r.qualifying_date,
            "circuit": circuits.get(r.circuit_id, {}), "circuit_id": r.circuit_id, "laps": int(r.laps),
            "length_km": float(r.course_length), "turns": int(r.turns),
            "drivers": [{
                "id": x.driver_id, "team": x.constructor_id, "p_win": round(float(x.p_win), 4),
                "p_pole": round(float(x.p_pole), 4), "p_podium": round(float(x.p_podium), 4),
                "circ_starts": int(x.d_circ_starts), "circ_wins": int(x.d_circ_wins),
                "circ_avg_finish": (round(float(x.d_circ_avg_finish), 1) if pd.notna(x.d_circ_avg_finish) else None),
                "pts_last5": round(float(x.d_pts_last5), 1),
            } for x in g.itertuples()],
            "strategy": circuit_profile(d, r.circuit_id),
        })

    # ---- standings and title maths ----
    ds = d["driver_standings"].sort_values("position")
    remaining = len(upcoming_out)
    sprints_left = int(d["races"][(d["races"].season == season) & ~d["races"].completed].is_sprint_weekend.sum())
    max_left = remaining * 25 + sprints_left * 8
    leader, second = ds.iloc[0], ds.iloc[1]
    title = {
        "leader": leader.driverId, "leader_pts": float(leader.points), "second": second.driverId,
        "second_pts": float(second.points), "gap": float(leader.points - second.points),
        "races_left": remaining, "max_points_left": max_left,
        "alive": [r.driverId for r in ds.itertuples() if float(r.points) + max_left >= float(leader.points)],
        "clinch_scenarios": _clinch_scenarios(ds, form, max_left, remaining),
    }

    names = {k: drivers[k] for k in set(up.driver_id) | set(ds.driverId) | set(sum(([h["pole"], h["winner"]] for u in upcoming_out for h in u["strategy"]["history"]), [])) | {r[k] for r in record for k in ("pick", "winner", "pole_pick", "pole") if r[k]} | {p[k] for p in season_poles(d, season) for k in ("pole", "winner") if p[k]} if k in drivers}
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "season": season,
        "data_through": {"round": int(races[races.completed & (races.season == season)]["round"].max()),
                         "gp": races[races.completed & (races.season == season)].sort_values("round").grand_prix_id.iloc[-1]},
        "drivers": names,
        "teams": {k: {"name": teams.get(k, k), "color": TEAM_COLORS.get(k, "#888888")} for k in set(up.constructor_id) | {h[t] for u in upcoming_out for h in u["strategy"]["history"] for t in ("pole_team", "winner_team") if h[t]}},
        "grands_prix": {k: v.get("short_name") or v["name"] for k, v in gps.items()},
        "circuits": {k: v["name"] for k, v in circuits.items()},
        "upcoming": upcoming_out,
        "standings": [{"pos": int(r.position), "driver": r.driverId, "points": float(r.points)} for r in ds.itertuples()],
        "constructor_standings": [{"pos": int(r.position), "team": r.constructorId, "points": float(r.points)} for r in d["constructor_standings"].sort_values("position").itertuples()],
        "title": title,
        "season_poles": season_poles(d, season),
        "form": form,
        "track_record": record,
        "evaluation": evaluation,
        "importance": importance,
        "n_train_races": int(train.race_id.nunique()),
        "train_seasons": [int(train.season.min()), int(train.season.max())],
    }
    (ROOT / "dashboard").mkdir(exist_ok=True)
    path = ROOT / "dashboard" / "data.json"
    path.write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    for u in upcoming_out:
        top = u["drivers"][:3]
        print(f"R{u['round']} {u['gp']}: " + ", ".join(f"{x['id']} {x['p_win']:.0%}" for x in top),
              "| pole:", max(u["drivers"], key=lambda x: x["p_pole"])["id"])


if __name__ == "__main__":
    main()
