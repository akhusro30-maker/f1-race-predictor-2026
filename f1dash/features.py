"""Pre-qualifying features: everything is known before the race weekend starts.

For every race (completed or upcoming) and every driver entered, features are built only
from races that happened strictly earlier, so the backtest has no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DNF_SCORE = 21.0  # finishing "position" used for non-classified results


def _driver_race_table(d: dict, first_season: int) -> pd.DataFrame:
    races = d["races"]
    races = races[races.season >= first_season - 1].copy()
    res = d["results"].merge(d["quali"][["race_id", "driver_id", "q_pos", "q_best"]], on=["race_id", "driver_id"], how="left")
    res = res.merge(races[["race_id", "date", "season", "round", "circuit_id"]], on="race_id")
    # pole time per race, gap to pole in %
    pole = d["quali"].groupby("race_id").q_best.min().rename("pole_time")
    res = res.merge(pole, on="race_id", how="left")
    res["q_gap_pct"] = (res.q_best / res.pole_time - 1) * 100
    res["q_gap_pct"] = res.q_gap_pct.clip(upper=7)  # cap outliers (wet / no time set)
    res["finish"] = res.position.where(res.classified, DNF_SCORE).astype(float)
    res["win"] = (res.position == 1).astype(int)
    res["podium"] = (res.position <= 3).astype(int)
    res["pole"] = (res.q_pos == 1).astype(int)
    res["dnf"] = (~res.classified).astype(int)
    res["pos_gain"] = (res.grid - res.position).where(res.classified & res.grid.notna())
    return res.sort_values(["date", "race_id"]).reset_index(drop=True)


def _history_features(hist: pd.DataFrame, drivers: list[tuple[str, str]], race: pd.Series) -> pd.DataFrame:
    """Features for one race from `hist` (all results strictly before it)."""
    rows = []
    season_hist = hist[hist.season == race.season]
    season_pts = season_hist.groupby("driver_id").points.sum()
    team_season_pts = season_hist.groupby("constructor_id").points.sum()
    # team pace: per race, best quali gap among the two cars; mean over last 5 races the team entered
    team_race = hist.groupby(["constructor_id", "race_id", "date"], as_index=False).agg(
        t_gap=("q_gap_pct", "min"), t_qpos=("q_pos", "min"), t_pts=("points", "sum"), t_finish=("finish", "min"))
    team_race = team_race.sort_values("date")
    circ = hist[hist.circuit_id == race.circuit_id]

    for driver_id, team_id in drivers:
        h = hist[hist.driver_id == driver_id]
        h_last = h.tail(10)
        h5 = h.tail(5)
        t = team_race[team_race.constructor_id == team_id].tail(5)
        # teammate head-to-head in quali over the last 8 shared races
        tm = hist[(hist.constructor_id == team_id) & (hist.race_id.isin(h.tail(8).race_id))]
        tm_cmp = []
        for rid, g in tm.groupby("race_id"):
            me = g[g.driver_id == driver_id].q_pos
            other = g[g.driver_id != driver_id].q_pos
            if len(me) and len(other) and me.notna().all() and other.notna().any():
                tm_cmp.append(float(other.min() - me.iloc[0]))
        dc = circ[circ.driver_id == driver_id].tail(5)
        ewm = lambda s, span=4: float(s.ewm(span=span).mean().iloc[-1]) if len(s.dropna()) else np.nan  # noqa: E731
        rows.append(
            {
                "driver_id": driver_id,
                "constructor_id": team_id,
                "d_starts": len(h),
                "d_finish_ewm": ewm(h.finish.tail(12)),
                "d_qpos_ewm": ewm(h.q_pos.dropna().tail(12)),
                "d_qgap_ewm": ewm(h.q_gap_pct.dropna().tail(12)),
                "d_pts_last5": h5.points.mean() if len(h5) else 0.0,
                "d_win_rate10": h_last.win.mean() if len(h_last) else 0.0,
                "d_podium_rate10": h_last.podium.mean() if len(h_last) else 0.0,
                "d_pole_rate10": h_last.pole.mean() if len(h_last) else 0.0,
                "d_dnf_rate10": h_last.dnf.mean() if len(h_last) else 0.2,
                "d_season_pts": float(season_pts.get(driver_id, 0.0)),
                "d_tm_quali_delta": float(np.mean(tm_cmp)) if tm_cmp else 0.0,
                "d_circ_starts": len(dc),
                "d_circ_avg_finish": dc.finish.mean() if len(dc) else np.nan,
                "d_circ_wins": int(dc.win.sum()),
                "t_gap_last5": t.t_gap.mean() if len(t) else np.nan,
                "t_qpos_last5": t.t_qpos.mean() if len(t) else np.nan,
                "t_pts_last5": t.t_pts.mean() if len(t) else 0.0,
                "t_best_finish_last5": t.t_finish.mean() if len(t) else np.nan,
                "t_season_pts": float(team_season_pts.get(team_id, 0.0)),
            }
        )
    f = pd.DataFrame(rows)
    # circuit character: how much the order changes, and how often pole wins (last 10 visits)
    cr = circ[circ.race_id.isin(circ.race_id.drop_duplicates().tail(10))]
    f["c_mean_abs_gain"] = cr.pos_gain.abs().mean() if len(cr) else np.nan
    winners = cr[cr.win == 1]
    f["c_pole_win_rate"] = (winners.grid == 1).mean() if len(winners) else np.nan
    f["round_frac"] = race["round"] / 24
    # relative-to-field versions: the target is relative, so these carry most of the signal
    for col, asc in [("d_finish_ewm", True), ("d_qpos_ewm", True), ("d_qgap_ewm", True), ("d_pts_last5", False),
                     ("d_season_pts", False), ("t_gap_last5", True), ("t_pts_last5", False), ("t_season_pts", False)]:
        f[col + "_rank"] = f[col].rank(ascending=asc, method="min", na_option="bottom")
    f["t_gap_vs_best"] = f.t_gap_last5 - f.t_gap_last5.min()
    f["d_season_pts_share"] = f.d_season_pts / max(f.d_season_pts.max(), 1)
    f["d_pts_last5_vs_best"] = f.d_pts_last5 - f.d_pts_last5.max()
    return f


FEATURES = [
    "d_starts", "d_finish_ewm", "d_qpos_ewm", "d_qgap_ewm", "d_pts_last5", "d_win_rate10", "d_podium_rate10",
    "d_pole_rate10", "d_dnf_rate10", "d_tm_quali_delta", "d_circ_starts", "d_circ_avg_finish", "d_circ_wins",
    "t_gap_last5", "t_qpos_last5", "t_pts_last5", "t_best_finish_last5", "c_mean_abs_gain", "c_pole_win_rate",
    "round_frac", "d_finish_ewm_rank", "d_qpos_ewm_rank", "d_qgap_ewm_rank", "d_pts_last5_rank", "d_season_pts_rank",
    "t_gap_last5_rank", "t_pts_last5_rank", "t_season_pts_rank", "t_gap_vs_best", "d_season_pts_share",
    "d_pts_last5_vs_best",
]


def build(d: dict, first_season: int = 2014) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_frame for completed races, upcoming_frame for races without results)."""
    table = _driver_race_table(d, first_season)
    races = d["races"][d["races"].season >= first_season]
    out = []
    upcoming_entries = None
    for _, race in races.iterrows():
        hist = table[table.date < race.date]
        if race.completed:
            entries = table[table.race_id == race.race_id][["driver_id", "constructor_id"]]
        else:
            if upcoming_entries is None:  # entry list from the most recent completed race
                last = table[table.date == table.date.max()]
                upcoming_entries = last[["driver_id", "constructor_id"]]
            entries = upcoming_entries
        f = _history_features(hist, list(entries.itertuples(index=False, name=None)), race)
        f["race_id"] = race.race_id
        f["season"] = race.season
        f["round"] = race["round"]
        f["completed"] = race.completed
        if race.completed:
            tgt = table[table.race_id == race.race_id][["driver_id", "win", "podium", "pole", "finish", "points"]]
            f = f.merge(tgt, on="driver_id", how="left")
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    return df[df.completed].reset_index(drop=True), df[~df.completed].reset_index(drop=True)


if __name__ == "__main__":
    import time

    from .load import load

    t = time.time()
    d = load()
    tr, up = build(d)
    print(tr.shape, up.shape, f"{time.time() - t:.1f}s")
    print(tr.groupby("race_id").win.sum().value_counts())
    print(up.groupby("race_id").size())
    tr.to_pickle("/tmp/claude-0/-home-claude/caab8b6d-c96a-52eb-818c-0e71167938ab/scratchpad/train.pkl")
    up.to_pickle("/tmp/claude-0/-home-claude/caab8b6d-c96a-52eb-818c-0e71167938ab/scratchpad/upcoming.pkl")
