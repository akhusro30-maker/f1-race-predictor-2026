"""Circuit strategy profile from historical pit stops, grids and results."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pirelli's dry compound picks (hard / medium / soft). 2026 values from Pirelli's 10 Oct 2026 announcement;
# None means not announced yet when this was built, so the previous year's choice is shown for reference.
COMPOUNDS_2026 = {
    "mexico-city": {"set": "C3 / C4 / C5", "note": "No gap between compounds this year (2025 ran C2 hard). The hard is softer, so it is a real race option rather than a tyre teams avoid."},
    "interlagos": {"set": "C3 / C4 / C5", "note": "One step softer than 2025, when the hard was barely raced and most teams two-stopped on the softer tyres. Expect more stops, possibly all three compounds."},
    "las-vegas": {"set": None, "prev": "C3 / C4 / C5 (2025)", "note": "Cold track temperatures make tyre warm-up, not wear, the main problem."},
    "lusail": {"set": None, "prev": "C1 / C2 / C3 (2025)", "note": "In 2023 and 2025 a maximum stint length was imposed for tyre safety, which forced extra stops. Check whether a limit applies again."},
    "yas-marina": {"set": None, "prev": "C3 / C4 / C5 (2025)", "note": "Low degradation track; a one-stop is usually fastest."},
}


def circuit_profile(d: dict, circuit_id: str, recent_from: int = 2022) -> dict:
    races = d["races"]
    cr = races[(races.circuit_id == circuit_id) & races.completed]
    res = d["results"].merge(cr[["race_id", "season", "laps"]].rename(columns={"laps": "race_laps"}), on="race_id")
    recent = res[res.season >= recent_from]
    pits = d["pits"].merge(cr[["race_id", "season", "laps"]].rename(columns={"laps": "race_laps"}), on="race_id")
    pits = pits[pits.season >= recent_from]

    out: dict = {"circuit_id": circuit_id, "editions_recent": sorted(recent.season.unique().tolist())}

    # --- stops: classified finishers only (a DNF's stop count is meaningless) ---
    fin = recent[recent.classified][["race_id", "season", "driver_id", "position", "grid", "race_laps"]]
    nstops = pits.groupby(["race_id", "driver_id"]).size().rename("stops")
    fin = fin.merge(nstops, on=["race_id", "driver_id"], how="left").fillna({"stops": 0})
    fin["stops"] = fin.stops.clip(upper=3).astype(int)
    if len(fin):
        dist = fin.stops.value_counts(normalize=True).sort_index()
        out["stop_share"] = {int(k): round(float(v), 3) for k, v in dist.items()}
        top10 = fin[fin.position <= 10]
        out["stop_share_top10"] = {int(k): round(float(v), 3) for k, v in top10.stops.value_counts(normalize=True).sort_index().items()}
        out["winner_stops"] = [{"season": int(r.season), "stops": int(r.stops)} for r in fin[fin.position == 1].itertuples()]
        fin["gain"] = fin.grid - fin.position
        g = fin[fin.grid.notna()].groupby("stops").agg(avg_gain=("gain", "mean"), avg_finish=("position", "mean"), n=("gain", "size"))
        out["by_stops"] = [{"stops": int(i), **{k: round(float(v), 2) for k, v in r.items()}} for i, r in g.iterrows() if r.n >= 8]
        # most common plan in the top 10, weighting the most recent edition double
        w = np.where(top10.season == top10.season.max(), 2.0, 1.0)
        out["typical_stops"] = int(pd.Series(w, index=top10.index).groupby(top10.stops).sum().idxmax())

    # --- pit windows (laps) for one- and two-stoppers in the top 10 ---
    if len(pits):
        p = pits.merge(fin[["race_id", "driver_id", "stops", "position"]], on=["race_id", "driver_id"])
        p = p[p.position <= 10]
        laps = int(cr.laps.iloc[-1]) if len(cr) else None
        win = {}
        for k in (1, 2):
            pk = p[p.stops == k]
            for s in range(1, k + 1):
                x = pk[pk.stop == s]
                if len(x) >= 6 and x.race_id.nunique() >= 2:
                    frac = x.lap / x.race_laps
                    win[f"{k}stop_{s}"] = [int(round(frac.quantile(0.25) * laps)), int(round(frac.median() * laps)),
                                           int(round(frac.quantile(0.75) * laps))]
        out["windows"] = win
        dur = pits.duration[(pits.duration > 10) & (pits.duration < 40)]
        out["pit_lane_time"] = round(float(dur.median()), 1) if len(dur) else None

    # --- qualifying importance (2014+ so there is enough history) ---
    long = d["results"].merge(cr[["race_id", "season"]], on="race_id")
    long = long[long.season >= 2014]
    winners = long[long.position == 1]
    out["pole_win_rate"] = round(float((winners.grid == 1).mean()), 3) if len(winners) else None
    out["front_row_win_rate"] = round(float((winners.grid <= 2).mean()), 3) if len(winners) else None
    out["winner_avg_grid"] = round(float(winners.grid.mean()), 2) if len(winners) else None
    out["deepest_winner"] = (int(winners.grid.max()), int(winners.loc[winners.grid.idxmax(), "season"])) if len(winners) else None
    cl = long[long.classified & long.grid.notna()]
    out["mean_abs_change"] = round(float((cl.grid - cl.position).abs().mean()), 2) if len(cl) else None
    out["n_editions_2014"] = int(winners.season.nunique())

    # --- past pole sitters and winners here ---
    q = d["quali"].merge(cr[["race_id", "season"]], on="race_id")
    poles = q[q.q_pos == 1].sort_values("season", ascending=False).head(8)
    wins = d["results"].merge(cr[["race_id", "season"]], on="race_id")
    wins = wins[wins.position == 1].set_index("season")
    out["history"] = [
        {"season": int(r.season), "pole": r.driver_id, "pole_team": r.constructor_id,
         "pole_time": r.q_best,
         "winner": wins.driver_id.get(r.season), "winner_team": wins.constructor_id.get(r.season),
         "winner_grid": int(wins.grid.get(r.season)) if pd.notna(wins.grid.get(r.season)) else None}
        for r in poles.itertuples()
    ]
    out["compounds"] = COMPOUNDS_2026.get(circuit_id)
    out["recommendation"] = _recommend(out)
    return out


def _recommend(p: dict) -> dict:
    """Plain-language strategy call from the numbers above."""
    stops = p.get("typical_stops", 1)
    comp = p.get("compounds") or {}
    cid = p["circuit_id"]
    # 2026 compound shifts nudge the plan
    if cid == "interlagos" and stops == 1:
        stops = 2
    plan = {1: "One-stop", 2: "Two-stop", 3: "Three-stop"}.get(stops, f"{stops}-stop")
    w = p.get("windows", {})
    if cid == "lusail":
        detail = ("Recent Qatar races had a mandatory maximum stint length, so stop timing was set by the rule. "
                  "If a limit applies again, plan stops to the limit; without one, two stops is the expected plan.")
    elif stops == 1 and "1stop_1" in w:
        lo, mid, hi = w["1stop_1"]
        detail = f"Pit around lap {mid} (top-10 runners stopped between laps {lo} and {hi})."
    elif stops == 2 and "2stop_1" in w and "2stop_2" in w:
        detail = f"Stops around laps {w['2stop_1'][1]} and {w['2stop_2'][1]}."
    else:
        detail = "Not enough recent stop data at this circuit to pin down a window."
    pw = p.get("pole_win_rate") or 0
    mac = p.get("mean_abs_change") or 0
    fr = p.get("front_row_win_rate") or 0
    small = (p.get("n_editions_2014") or 0) < 5
    if pw >= 0.55 or fr >= 0.85 or mac < 2.5:
        quali = "Track position is everything here: qualifying decides most races, so an undercut is the main way to gain places."
        if small and pw < 0.55:
            quali = (f"Every winner here has started from the front row ({p['n_editions_2014']} races so far), though not always "
                     "from pole. Qualifying in the top two matters more than pole itself.")
    elif pw >= 0.35:
        quali = "Qualifying matters, but winners do come from the second and third rows, so a strong race car can recover."
    else:
        quali = "Pole is a weak advantage here; overtaking and strategy decide far more than grid position."
    alt = None
    bs = {b["stops"]: b for b in p.get("by_stops", [])}
    if 1 in bs and 2 in bs:
        better = 2 if bs[2]["avg_gain"] > bs[1]["avg_gain"] else 1
        alt = (f"Recently, {('two' if better == 2 else 'one')}-stoppers gained more places on average "
               f"({bs[better]['avg_gain']:+.1f} vs {bs[3 - better]['avg_gain']:+.1f}).")
    return {"plan": plan, "detail": detail, "quali": quali, "alt": alt, "compound_note": comp.get("note")}


def season_poles(d: dict, season: int) -> list[dict]:
    races = d["races"][(d["races"].season == season) & d["races"].completed]
    q = d["quali"][d["quali"].q_pos == 1].set_index("race_id")
    r = d["results"][d["results"].position == 1].set_index("race_id")
    out = []
    for x in races.itertuples():
        out.append({
            "round": int(x.round), "gp": x.grand_prix_id, "circuit": x.circuit_id, "date": str(x.date.date()),
            "pole": q.driver_id.get(x.race_id), "pole_team": q.constructor_id.get(x.race_id),
            "winner": r.driver_id.get(x.race_id), "winner_team": r.constructor_id.get(x.race_id),
            "winner_grid": int(r.grid.get(x.race_id)) if pd.notna(r.grid.get(x.race_id)) else None,
        })
    return out
