"""Load F1DB YAML data (https://github.com/f1db/f1db) into flat pandas tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as Loader

DATA = Path(__file__).resolve().parents[2] / "f1db" / "f1db" / "src" / "data"  # clone of github.com/f1db/f1db next to this project


def _y(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=Loader)


def _laptime(s) -> float | None:
    """'1:34.035' -> 94.035 seconds."""
    if not s:
        return None
    s = str(s)
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return None


def load(first_season: int = 1990, data_dir: Path = DATA) -> dict[str, pd.DataFrame]:
    races, results, quali, grids, pits = [], [], [], [], []
    seasons = sorted(int(p.name) for p in (data_dir / "seasons").iterdir() if p.name.isdigit())
    for season in seasons:
        if season < first_season:
            continue
        for rdir in sorted((data_dir / "seasons" / str(season) / "races").iterdir()):
            r = _y(rdir / "race.yml")
            rid = r["id"]
            has_result = (rdir / "race-results.yml").exists()
            races.append(
                {
                    "race_id": rid,
                    "season": season,
                    "round": r["round"],
                    "date": pd.Timestamp(str(r["date"])),
                    "grand_prix_id": r["grandPrixId"],
                    "official_name": r.get("officialName"),
                    "circuit_id": r["circuitId"],
                    "laps": r.get("laps"),
                    "distance": r.get("distance"),
                    "turns": r.get("turns"),
                    "course_length": r.get("courseLength"),
                    "qualifying_date": str(r.get("qualifyingDate") or ""),
                    "is_sprint_weekend": bool(r.get("sprintRaceDate")),
                    "completed": has_result,
                }
            )
            for row in _y(rdir / "race-results.yml") or []:
                pos = row.get("position")
                results.append(
                    {
                        "race_id": rid,
                        "driver_id": row["driverId"],
                        "constructor_id": row["constructorId"],
                        "position": int(pos) if isinstance(pos, int) else None,
                        "status": "FIN" if isinstance(pos, int) else str(pos),
                        "classified": isinstance(pos, int),
                        "laps": row.get("laps"),
                        "points": float(row.get("points") or 0),
                        "grid": row.get("gridPosition") if isinstance(row.get("gridPosition"), int) else None,
                        "reason_retired": row.get("reasonRetired"),
                    }
                )
            for row in _y(rdir / "qualifying-results.yml") or []:
                pos = row.get("position")
                times = [t for t in (_laptime(row.get(k)) for k in ("q1", "q2", "q3", "time")) if t]
                quali.append(
                    {
                        "race_id": rid,
                        "driver_id": row["driverId"],
                        "constructor_id": row["constructorId"],
                        "q_pos": int(pos) if isinstance(pos, int) else None,
                        "q_best": min(times) if times else None,
                    }
                )
            for row in _y(rdir / "starting-grid-positions.yml") or []:
                pos = row.get("position")
                grids.append(
                    {
                        "race_id": rid,
                        "driver_id": row["driverId"],
                        "grid_pos": int(pos) if isinstance(pos, int) else None,
                    }
                )
            for row in _y(rdir / "pit-stops.yml") or []:
                pits.append(
                    {
                        "race_id": rid,
                        "driver_id": row["driverId"],
                        "stop": row.get("stop"),
                        "lap": row.get("lap"),
                        "duration": _laptime(row.get("time")),
                    }
                )

    drivers = {}
    for p in (data_dir / "drivers").glob("*.yml"):
        d = _y(p)
        drivers[d["id"]] = {
            "name": d["name"],
            "last_name": d.get("lastName"),
            "abbr": d.get("abbreviation"),
            "number": d.get("permanentNumber"),
            "nationality": d.get("nationalityCountryId"),
        }
    constructors = {p.stem: _y(p)["name"] for p in (data_dir / "constructors").glob("*.yml")}
    circuits = {}
    for p in (data_dir / "circuits").glob("*.yml"):
        c = _y(p)
        circuits[c["id"]] = {"name": c["name"], "full_name": c.get("fullName"), "place": c.get("placeName"),
                             "country": c.get("countryId")}
    gps = {}
    for p in (data_dir / "grands-prix").glob("*.yml"):
        g = _y(p)
        gps[g["id"]] = {"name": g.get("name"), "full_name": g.get("fullName"), "short_name": g.get("shortName")}

    standings = _y(data_dir / "seasons" / str(seasons[-1]) / "driver-standings.yml") or []
    cstandings = _y(data_dir / "seasons" / str(seasons[-1]) / "constructor-standings.yml") or []

    return {
        "races": pd.DataFrame(races).sort_values(["date", "round"]).reset_index(drop=True),
        "results": pd.DataFrame(results),
        "quali": pd.DataFrame(quali),
        "grids": pd.DataFrame(grids),
        "pits": pd.DataFrame(pits),
        "drivers": drivers,
        "constructors": constructors,
        "circuits": circuits,
        "grands_prix": gps,
        "driver_standings": pd.DataFrame(standings),
        "constructor_standings": pd.DataFrame(cstandings),
    }


if __name__ == "__main__":
    import time

    t = time.time()
    d = load()
    for k, v in d.items():
        print(k, getattr(v, "shape", len(v)))
    print(f"{time.time() - t:.1f}s")
