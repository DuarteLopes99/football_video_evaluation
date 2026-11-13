"""Train the xG model on StatsBomb open data and write models/xg_model.json.

Data: https://github.com/statsbomb/open-data (CC BY-NC-SA, attribution "Data: StatsBomb").
Step 1 (once, ~1 GB download, then deleted): extract shots to a jsonl with
    python scripts/train_xg.py --extract --sb-dir /path/to/open-data
  where /path/to/open-data is a clone (sparse checkout of data/events is enough), or with
  `pip install statsbombpy` (``--statsbombpy``).
Step 2: fit + evaluate
    python scripts/train_xg.py --shots data/xg/shots_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_eval.xg import BASIC, FULL, feature_row, freeze_frame_features  # noqa: E402

COMPETITIONS = [(43, 106), (55, 43), (55, 282), (9, 281), (1267, 107), (7, 235), (7, 108), (11, 90), (44, 107)]


def sb_to_m(loc):
    """StatsBomb 120x80 yd (shooting team attacks +x) -> 105x68 m."""
    return loc[0] * 105 / 120, loc[1] * 68 / 80


def extract(sb_dir: str | None, use_statsbombpy: bool, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if use_statsbombpy:
        from statsbombpy import sb
        for comp, season in COMPETITIONS:
            for mid in sb.matches(competition_id=comp, season_id=season).match_id:
                ev = sb.events(match_id=mid, split=True, flatten_attrs=False)["shots"]
                for _, e in ev.iterrows():
                    s = e["shot"]
                    rows.append(_row(int(mid), e["location"], s))
    else:
        base = Path(sb_dir) / "data"
        for comp, season in COMPETITIONS:
            for m in json.load(open(base / "matches" / str(comp) / f"{season}.json")):
                p = base / "events" / f"{m['match_id']}.json"
                if not p.exists():
                    continue
                for e in json.load(open(p)):
                    if e["type"]["name"] == "Shot":
                        rows.append(_row(m["match_id"], e["location"], e["shot"], e.get("play_pattern", {}).get("name")))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} shots -> {out}")


def _row(mid, loc, s, play=None):
    return {"match_id": int(mid), "loc": loc, "body": s["body_part"]["name"], "type": s["type"]["name"],
            "outcome": s["outcome"]["name"], "sb_xg": s.get("statsbomb_xg"), "play": play,
            "ff": [{"loc": p["location"], "tm": p["teammate"], "pos": p["position"]["name"]}
                   for p in s.get("freeze_frame", []) or []]}


def build_table(shots_path: Path) -> pd.DataFrame:
    recs = []
    for line in open(shots_path):
        s = json.loads(line)
        x, y = sb_to_m(s["loc"])
        ff = None
        if s["ff"]:
            opp = [sb_to_m(p["loc"]) for p in s["ff"] if not p["tm"] and p["pos"] != "Goalkeeper"]
            gk = [sb_to_m(p["loc"]) for p in s["ff"] if not p["tm"] and p["pos"] == "Goalkeeper"]
            ff = freeze_frame_features((x, y), np.array(opp), gk[0] if gk else None)
        r = feature_row(x, y, ff)
        r.update(match_id=s["match_id"], goal=int(s["outcome"] == "Goal"), type=s["type"], body=s["body"],
                 sb_xg=s["sb_xg"], has_ff=ff is not None)
        recs.append(r)
    return pd.DataFrame(recs)


def fit_lr(df: pd.DataFrame, feats, C=1.0):
    X = df[feats].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, df.goal)
    return {"features": feats, "mean": mu.tolist(), "std": sd.tolist(), "coef": m.coef_[0].tolist(),
            "intercept": float(m.intercept_[0])}


def predict(params, df):
    z = (df[params["features"]].to_numpy(float) - np.array(params["mean"])) / np.array(params["std"])
    return 1 / (1 + np.exp(-(params["intercept"] + z @ np.array(params["coef"]))))


def cv_eval(df, feats, k=5):
    p = np.zeros(len(df))
    for tr, te in GroupKFold(k).split(df, groups=df.match_id):
        p[te] = predict(fit_lr(df.iloc[tr], feats), df.iloc[te])
    return dict(auc=roc_auc_score(df.goal, p), brier=brier_score_loss(df.goal, p), logloss=log_loss(df.goal, p),
                mean_pred=float(p.mean()), goal_rate=float(df.goal.mean())), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=str(ROOT / "data/xg/shots_raw.jsonl"))
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--sb-dir")
    ap.add_argument("--statsbombpy", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "models/xg_model.json"))
    a = ap.parse_args()
    if a.extract:
        extract(a.sb_dir, a.statsbombpy, Path(a.shots))
    df = build_table(Path(a.shots))
    pens = df[df.type == "Penalty"]
    fks = df[df.type == "Free Kick"]
    op = df[df.type == "Open Play"].reset_index(drop=True)
    opf = op[op.has_ff].reset_index(drop=True)
    report = {"n_shots": len(df), "n_open_play": len(op), "n_open_play_ff": len(opf)}
    for name, d, feats in (("basic", op, BASIC), ("full", opf, FULL)):
        m, _ = cv_eval(d, feats)
        sb = d.dropna(subset=["sb_xg"])
        m["statsbomb_xg_auc_same_shots"] = roc_auc_score(sb.goal, sb.sb_xg)
        m["statsbomb_xg_brier_same_shots"] = brier_score_loss(sb.goal, sb.sb_xg)
        report[name] = m
    params = {"basic": fit_lr(op, BASIC), "full": fit_lr(opf, FULL),
              "penalty_xg": float(pens.goal.mean()) if len(pens) else 0.76,
              "dfk_factor": float(fks.goal.mean() / max(predict(fit_lr(op, BASIC), fks).mean(), 1e-6)) if len(fks) else 1.0,
              "training": {"source": "StatsBomb open data", "competitions": COMPETITIONS, "report": report}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(report, indent=2))
    print("->", a.out)


if __name__ == "__main__":
    main()
