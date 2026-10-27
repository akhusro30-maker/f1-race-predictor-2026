"""Win and pole models, walk-forward backtest and Plackett-Luce podium probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

RECENCY = 0.8  # each older season counts 0.8x the next (regulations change, old form ages)


def _weights(seasons: pd.Series, target_season: int) -> np.ndarray:
    return RECENCY ** (target_season - seasons.to_numpy())


def _norm(df: pd.DataFrame, raw: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Rescale per-driver scores so they sum to `total` within each race (1 winner, 3 podium places)."""
    s = pd.Series(np.clip(raw, 1e-6, None), index=df.index)
    p = s / s.groupby(df.race_id).transform("sum") * total
    if total > 1:  # a probability cannot exceed 1: cap and hand the excess to the rest of the field
        for _ in range(10):
            over = (p - 0.97).clip(lower=0)
            if over.sum() < 1e-9:
                break
            p = p.clip(upper=0.97)
            room = (0.97 - p).clip(lower=0)
            p = p + room / room.groupby(df.race_id).transform("sum") * over.groupby(df.race_id).transform("sum")
    return p.to_numpy()


class RaceModel:
    """Blend of a regularised logistic regression and a small LightGBM, normalised per race."""

    def __init__(self, target: str = "win"):
        self.target = target
        self.total = 3.0 if target == "podium" else 1.0

    def fit(self, train: pd.DataFrame, target_season: int):
        X, y = train[FEATURES], train[self.target].astype(int)
        w = getattr(self, "recency", RECENCY) ** (target_season - train.season.to_numpy())
        self.lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.2, max_iter=2000))
        self.lr.fit(X, y, logisticregression__sample_weight=w)
        self.gbm = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                                  verbose=-1, random_state=7)
        self.gbm.fit(X, y, sample_weight=w)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        p_lr = _norm(df, self.lr.predict_proba(X)[:, 1], self.total)
        p_gbm = _norm(df, self.gbm.predict_proba(X)[:, 1], self.total)
        p = _norm(df, 0.5 * p_lr + 0.5 * p_gbm, self.total)
        return pd.DataFrame({"p_lr": p_lr, "p_gbm": p_gbm, "p": p}, index=df.index)

    def importance(self) -> pd.Series:
        g = pd.Series(self.gbm.booster_.feature_importance("gain"), index=FEATURES)
        return (g / g.sum()).sort_values(ascending=False)


def plackett_luce(p_win: np.ndarray, n_sims: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
    """Simulate finishing orders where each next place is drawn in proportion to win strength."""
    rng = np.random.default_rng(seed)
    w = np.clip(np.asarray(p_win, float), 1e-9, None)
    # Gumbel trick: sorting log(w) + Gumbel noise samples a Plackett-Luce ordering
    keys = np.log(w)[None, :] + rng.gumbel(size=(n_sims, len(w)))
    order = np.argsort(-keys, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n_sims)[:, None], order] = np.arange(1, len(w) + 1)[None, :]
    return {"p_podium": (pos <= 3).mean(0), "p_points": (pos <= 10).mean(0), "exp_pos": pos.mean(0)}


def baseline_form(df: pd.DataFrame) -> np.ndarray:
    """Naive benchmark: chance proportional to average points over the last 5 races."""
    return _norm(df, df.d_pts_last5.to_numpy() + 0.25)


def walk_forward(train: pd.DataFrame, target: str, from_season: int = 2022, recency: float | None = None) -> pd.DataFrame:
    """Predict each race from from_season onwards using only earlier races."""
    races = train[train.season >= from_season][["race_id", "season", "round"]].drop_duplicates()
    out = []
    for _, r in races.iterrows():
        past = train[(train.season < r.season) | ((train.season == r.season) & (train["round"] < r["round"]))]
        cur = train[train.race_id == r.race_id]
        m = RaceModel(target)
        if recency is not None:
            m.recency = recency
        m.fit(past, r.season)
        pr = m.predict(cur)
        res = cur[["race_id", "season", "round", "driver_id", "constructor_id", target]].copy()
        res = res.join(pr)
        res["p_base"] = baseline_form(cur) * (3 if target == "podium" else 1)
        out.append(res)
    return pd.concat(out, ignore_index=True)


def score(bt: pd.DataFrame, target: str, cols=("p", "p_lr", "p_gbm", "p_base"), n_boot: int = 2000) -> dict:
    """Top-1 / top-3 hit rate and log loss of the actual winner, with bootstrap CIs over races."""
    rng = np.random.default_rng(1)
    out = {}
    per_race = {}
    for c in cols:
        g = bt.groupby("race_id")
        top1 = g.apply(lambda x: float(x.loc[x[c].idxmax(), target] == 1), include_groups=False)
        top3 = g.apply(lambda x: float(x.nlargest(3, c)[target].max() == 1), include_groups=False)
        ll = g.apply(lambda x: -np.log(max(x.loc[x[target] == 1, c].sum(), 1e-6)), include_groups=False)
        per_race[c] = pd.DataFrame({"top1": top1, "top3": top3, "ll": ll})
    races = per_race[cols[0]].index.to_numpy()
    for c in cols:
        pr = per_race[c]
        boot = [pr.loc[rng.choice(races, len(races))].mean() for _ in range(n_boot)]
        boot = pd.DataFrame(boot)
        out[c] = {k: (float(pr[k].mean()), float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975)))
                  for k in ("top1", "top3", "ll")}
    # paired comparison of the blend against the form baseline
    diff = per_race["p_base"].ll - per_race["p"].ll
    boot = [diff.loc[rng.choice(races, len(races))].mean() for _ in range(n_boot)]
    out["ll_improvement_vs_base"] = (float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    out["n_races"] = len(races)
    out["uniform_ll"] = float(np.log(bt.groupby("race_id").size()).mean())
    return out


def calibration(bt: pd.DataFrame, pcol: str, ycol: str, bins=(0, .05, .15, .3, .5, .7, 1.0)) -> list[dict]:
    b = pd.cut(bt[pcol], bins, include_lowest=True)
    g = bt.groupby(b, observed=True).agg(pred=(pcol, "mean"), actual=(ycol, "mean"), n=(ycol, "size"))
    return [{"bin": str(i), **{k: float(v) for k, v in r.items()}} for i, r in g.iterrows()]
