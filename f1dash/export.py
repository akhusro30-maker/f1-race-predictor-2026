"""Flatten dashboard/data.json into the row datasets the dashboard reads (dashboard/datasets/*.json)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = json.loads((ROOT / "dashboard" / "data.json").read_text())
    out = ROOT / "dashboard" / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    drv, teams, gps = d["drivers"], d["teams"], d["grands_prix"]
    name = lambda k: drv.get(k, {}).get("name", k) if k else None  # noqa: E731
    gpn = lambda k: (gps.get(k) or k.replace("-", " ").title()) if k else None  # noqa: E731
    sets = {}

    rows = []
    for u in d["upcoming"]:
        for i, x in enumerate(u["drivers"]):
            rows.append({
                "round": u["round"], "gp": gpn(u["gp"]), "driver_id": x["id"], "driver": name(x["id"]),
                "abbr": drv.get(x["id"], {}).get("abbr"), "team": teams.get(x["team"], {}).get("name", x["team"]),
                "team_color": teams.get(x["team"], {}).get("color"), "win_rank": i + 1,
                "p_win": x["p_win"], "p_pole": x["p_pole"], "p_podium": x["p_podium"],
                "circ_starts": x["circ_starts"], "circ_wins": x["circ_wins"], "circ_avg_finish": x["circ_avg_finish"],
                "pts_last5": x["pts_last5"],
            })
    sets["predictions"] = rows

    races, hist = [], []
    for u in d["upcoming"]:
        s, rec, c = u["strategy"], u["strategy"]["recommendation"], u["strategy"].get("compounds") or {}
        w = s.get("windows", {})
        sh = s.get("stop_share_top10", {})
        bs = {b["stops"]: b for b in s.get("by_stops", [])}
        win = lambda k, i: (w.get(k) or [None] * 3)[i]  # noqa: E731
        races.append({
            "round": u["round"], "gp": gpn(u["gp"]), "date": u["date"], "quali_date": u["quali_date"],
            "circuit": u["circuit"].get("name"), "place": u["circuit"].get("place"), "laps": u["laps"],
            "length_km": u["length_km"], "turns": u["turns"],
            "compounds": c.get("set") or "Not announced yet",
            "compounds_prev": c.get("prev"), "compound_note": rec.get("compound_note"),
            "plan": rec["plan"], "plan_detail": rec["detail"], "quali_note": rec["quali"], "alt_note": rec.get("alt"),
            "pole_win_rate": s.get("pole_win_rate"), "front_row_win_rate": s.get("front_row_win_rate"),
            "winner_avg_grid": s.get("winner_avg_grid"), "mean_abs_change": s.get("mean_abs_change"),
            "editions_since_2014": s.get("n_editions_2014"), "pit_lane_time": s.get("pit_lane_time"),
            "share_1stop": sh.get("1", sh.get(1, 0)), "share_2stop": sh.get("2", sh.get(2, 0)),
            "share_3stop": sh.get("3", sh.get(3, 0)),
            "gain_1stop": bs.get(1, {}).get("avg_gain"), "gain_2stop": bs.get(2, {}).get("avg_gain"),
            "gain_3stop": bs.get(3, {}).get("avg_gain"),
            "w1_lo": win("1stop_1", 0), "w1_mid": win("1stop_1", 1), "w1_hi": win("1stop_1", 2),
            "w2a_lo": win("2stop_1", 0), "w2a_mid": win("2stop_1", 1), "w2a_hi": win("2stop_1", 2),
            "w2b_lo": win("2stop_2", 0), "w2b_mid": win("2stop_2", 1), "w2b_hi": win("2stop_2", 2),
            "recent_seasons": ", ".join(str(x) for x in s.get("editions_recent", [])),
        })
        ws = {x["season"]: x["stops"] for x in s.get("winner_stops", [])}
        for h in s["history"]:
            hist.append({
                "round": u["round"], "season": h["season"], "pole": name(h["pole"]),
                "pole_team": teams.get(h["pole_team"], {}).get("name", h["pole_team"]),
                "winner": name(h["winner"]), "winner_grid": h["winner_grid"],
                "pole_won": "Yes" if h["pole"] == h["winner"] else "No",
                "winner_stops": ws.get(h["season"]),
            })
    sets["races"], sets["circuit_history"] = races, hist

    sets["season"] = [{
        "round": p["round"], "gp": gpn(p["gp"]), "date": p["date"], "pole": name(p["pole"]), "winner": name(p["winner"]),
        "winner_grid": p["winner_grid"], "pole_won": "Yes" if p["pole"] == p["winner"] else "No",
        **next(({"model_pick": name(r["pick"]), "pick_p": round(r["pick_p"], 4), "winner_p": round(r["winner_p"], 4),
                 "winner_rank": r["winner_rank"], "pick_won": "Yes" if r["hit"] else "No",
                 "pole_pick": name(r["pole_pick"]), "pole_pick_right": "Yes" if r["pole_hit"] else "No"}
                for r in d["track_record"] if r["round"] == p["round"]), {}),
    } for p in d["season_poles"]]

    st = []
    for s in d["standings"]:
        f = d["form"].get(s["driver"], {})
        t = f.get("team")
        st.append({"pos": s["pos"], "driver_id": s["driver"], "driver": name(s["driver"]),
                   "team": teams.get(t, {}).get("name", t), "team_color": teams.get(t, {}).get("color"),
                   "points": s["points"], "wins": f.get("wins", 0), "podiums": f.get("podiums", 0),
                   "poles": f.get("poles", 0), "dnfs": f.get("dnfs", 0),
                   "last5": " · ".join(str(r["pos"]) for r in f.get("results", [])[-5:])})
    sets["standings"] = st

    t = d["title"]
    sets["title"] = [{"leader": name(t["leader"]), "leader_pts": t["leader_pts"], "rival": name(c["rival"]),
                      "gap": c["gap"], "worst_finish_if_rival_wins": c["worst_finish_if_rival_wins"],
                      "races_left": t["races_left"], "max_points_left": t["max_points_left"]}
                     for c in t["clinch_scenarios"]]

    ev = d["evaluation"]
    ov = []
    for target, label in (("win", "Race winner"), ("pole", "Pole sitter")):
        e = ev[target]
        for metric, mlabel in (("top1", "Favourite was right"), ("top3", "Answer in top 3 picks"), ("ll", "Log loss (lower is better)")):
            m, b = e["overall"]["p"][metric], e["overall"]["p_base"][metric]
            ov.append({"id": f"{target}_{metric}", "target": label, "metric": mlabel, "model": round(m[0], 4),
                       "model_lo": round(m[1], 4), "model_hi": round(m[2], 4), "baseline": round(b[0], 4),
                       "races": e["n_races"]})
        ov.append({"id": f"{target}_llgain", "target": label, "metric": "Log loss improvement vs baseline",
                   "model": round(e["ll_gain"][0], 4), "model_lo": round(e["ll_gain"][1], 4),
                   "model_hi": round(e["ll_gain"][2], 4), "baseline": 0, "races": e["n_races"]})
    ov.append({"id": "podium_brier", "target": "Podium", "metric": "Brier score (lower is better)",
               "model": round(ev["podium"]["brier"], 4), "model_lo": None, "model_hi": None,
               "baseline": round(ev["podium"]["brier_base"], 4), "races": ev["win"]["n_races"]})
    sets["backtest"] = ov
    sets["backtest_seasons"] = [{"season": s["season"], "races": s["races"], "model_top1": round(s["top1"], 4),
                                 "baseline_top1": round(s["top1_base"], 4)} for s in ev["win"]["by_season"]]
    sets["calibration"] = [{"model": lab, "bin": c["bin"], "predicted": round(c["pred"], 4), "actual": round(c["actual"], 4),
                            "n": int(c["n"])} for key, lab in (("win", "Win"), ("podium", "Podium"))
                           for c in ev[key]["calibration"]]
    sets["importance"] = [{"feature": x["feature"], "share": round(x["share"], 4)} for x in d["importance"]]

    for k, v in sets.items():
        (out / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False))
        print(k, len(v))


if __name__ == "__main__":
    main()
