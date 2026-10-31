"""Team and role assignment from shirt appearance.

``color`` backend: torso crop -> remove the *pitch colour of this frame* (:func:`grass_model`) -> descriptor = chroma
quartiles + hue histogram + white / black shares (:func:`color_descriptor`, no lightness: sun vs shade).
``siglip`` backend: SigLIP image embeddings (Roboflow `sports`: SigLIP -> UMAP -> KMeans(2)).

* **With the players clicked in the setup** (recommended): :class:`KitClassifier` — nearest clicked samples per class
  (both outfield kits, both goalkeepers, referee). Clicked boxes are first snapped to the detector box
  (:func:`snap_box`). The same classifier labels the live view (per-track running vote, :class:`TrackVotes`) and
  makes the final assignment; the tracks matched to a click on its frame keep the clicked class (:func:`apply_manual`).
* **Without clicks**: KMeans(k=2) fitted on outfield crops from the first ``fit_frames`` sampled frames.
  Crops far from both centroids (``outlier_quantile`` x ``outlier_factor``) are 'other' (referees,
  staff, sometimes goalkeepers). Goalkeepers are re-attached to a team later by pitch side
  (see :func:`assign_goalkeepers`). Final team per track = majority vote across its crops.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


def torso_crop(img: np.ndarray, box) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 10:
        return None
    cx0, cx1 = x0 + int(0.2 * w), x1 - int(0.2 * w)
    cy0, cy1 = y0 + int(0.15 * h), y0 + int(0.55 * h)
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(cx0, 0), max(cy0, 0), min(cx1, W), min(cy1, H)
    if cx1 - cx0 < 2 or cy1 - cy0 < 3:
        return None
    return img[cy0:cy1, cx0:cx1]


def grass_model(img: np.ndarray) -> Optional[np.ndarray]:
    """Median HSV of the playing surface in this frame (lower 2/3 of the image, green-ish pixels).

    Used to remove *only the pitch colour* from the shirt crops. The old fixed mask (hue 30-90 = "grass") also
    deleted green shirts — on the J3 video that made the green kit unrecognisable (13 / 43 clicked samples wrong).
    """
    hsv = cv2.cvtColor(img[img.shape[0] // 3::4, ::4], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    g = hsv[(hsv[:, 0] > 25) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40)]
    return np.median(g, 0).astype(float) if len(g) > 200 else None


def _grass_mask(hsv: np.ndarray, grass: Optional[np.ndarray]) -> np.ndarray:
    if grass is None:                                       # no estimate: generic green band
        return (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    dh = np.abs(hsv[..., 0].astype(int) - int(grass[0]))
    dh = np.minimum(dh, 180 - dh)
    return (dh < 9) & (np.abs(hsv[..., 1].astype(int) - int(grass[1])) < 70) & (hsv[..., 2] > 0.35 * grass[2])


DESC_DIM = 20


def color_descriptor(crop: np.ndarray, grass: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Shirt colour descriptor (20 dims): chroma (Lab a, b) quartiles, saturation-weighted hue histogram, share of
    white / black pixels. Lightness is left out on purpose (sun vs shade); white / black kits are still separated by
    the white / black shares. ``grass``: :func:`grass_model` of the frame (pixels of that colour are removed)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = ~_grass_mask(hsv, grass)
    if keep.sum() < 6:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(float)
    q = np.percentile(lab[:, 1:], [25, 50, 75], axis=0).ravel() / 255.0      # 6 dims
    h = hsv[..., 0][keep]
    s = hsv[..., 1][keep].astype(float)
    hist, _ = np.histogram(h, bins=12, range=(0, 180), weights=s / 255.0 + 0.05)
    hist = hist / (hist.sum() + 1e-9)
    v = hsv[..., 2][keep]
    bright = np.array([(v > 170).mean(), (v < 60).mean()])                    # white / black kits
    return np.r_[3.0 * q, hist, 1.5 * bright].astype(np.float32)


class SiglipEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, SiglipVisionModel
        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()

    def __call__(self, crops: List[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        with self.torch.no_grad():
            x = self.proc(images=rgb, return_tensors="pt").to(self.device)
            e = self.model(**x).pooler_output.cpu().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


class TeamClassifier:
    def __init__(self, backend: str = "color", outlier_quantile: float = 0.97, outlier_factor: float = 1.6,
                 device: str = "cpu"):
        self.backend = backend
        self.q, self.f = outlier_quantile, outlier_factor
        self.embed = SiglipEmbedder(device=device) if backend == "siglip" else None
        self.km: Optional[KMeans] = None
        self.thr = np.inf

    def describe(self, img: np.ndarray, boxes: np.ndarray) -> List[Optional[np.ndarray]]:
        crops = [torso_crop(img, b) for b in boxes]
        if self.embed is not None:
            ok = [i for i, c in enumerate(crops) if c is not None]
            out: List[Optional[np.ndarray]] = [None] * len(crops)
            if ok:
                E = self.embed([crops[i] for i in ok])
                for k, i in enumerate(ok):
                    out[i] = E[k]
            return out
        g = grass_model(img)
        return [color_descriptor(c, g) if c is not None else None for c in crops]

    def fit(self, feats: np.ndarray) -> "TeamClassifier":
        feats = np.asarray(feats)
        km3 = KMeans(3, n_init=10, random_state=0).fit(feats)
        # the two biggest clusters are the teams; the smallest one (referees, keepers) is dropped
        sizes = np.bincount(km3.labels_, minlength=3)
        big = np.argsort(sizes)[::-1][:2]
        core = np.isin(km3.labels_, big)
        self.km = KMeans(2, n_init=10, random_state=0).fit(feats[core])
        d = self.km.transform(feats[core]).min(1)
        self.thr = float(np.quantile(d, self.q) * self.f)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """0 / 1 = team, -1 = other (referee / staff / unknown)."""
        feats = np.asarray(feats)
        if len(feats) == 0:
            return np.zeros(0, int)
        d = self.km.transform(feats)
        lab = d.argmin(1)
        lab[d.min(1) > self.thr] = -1
        return lab


def vote(track_ids: np.ndarray, labels: np.ndarray, min_share: float = 0.6) -> dict:
    """Majority vote of crop labels per track id -> {track_id: team}."""
    out = {}
    for tid in np.unique(track_ids):
        if tid < 0:
            continue
        l = labels[track_ids == tid]
        vals, cnt = np.unique(l, return_counts=True)
        best = vals[cnt.argmax()]
        if cnt.max() / cnt.sum() >= min_share:
            out[int(tid)] = int(best)
        else:  # ambiguous: keep the most frequent *team* label if one exists
            team_mask = vals >= 0
            out[int(tid)] = int(vals[team_mask][cnt[team_mask].argmax()]) if team_mask.any() else -1
    return out


def assign_goalkeepers(tracks, pitch, team_side: dict, box_depth: float = 18.0, min_share: float = 0.7):
    """Tracks labelled 'other' that live inside one penalty area most of the time are that side's GK.

    ``team_side`` maps team -> True if the team defends the LEFT goal (x=0) in that period.
    ``tracks`` is the tracking DataFrame (needs track_id, team, x, y). Modified in place.
    """
    others = tracks[(tracks.team == -1) & tracks.x.notna() & (tracks.cls != "ball")]
    for tid, g in others.groupby("track_id"):
        if len(g) < 5:
            continue
        inside_l = ((g.x < box_depth) & (np.abs(g.y - pitch.width / 2) < 22)).mean()
        inside_r = ((g.x > pitch.length - box_depth) & (np.abs(g.y - pitch.width / 2) < 22)).mean()
        side_left = inside_l >= min_share
        side_right = inside_r >= min_share
        if side_left or side_right:
            team = [t for t, left in team_side.items() if left == side_left]
            if team:
                tracks.loc[tracks.track_id == tid, ["team", "cls"]] = [team[0], "goalkeeper"]
    return tracks


# ----------------------------------------------------------------------------- supervised: clicked kit samples
KIT_CLASSES = ("team0", "team1", "gk0", "gk1", "referee")
# label codes used in the per-crop predictions
CODE = {"team0": 0, "team1": 1, "gk0": 2, "gk1": 3, "referee": 4}
CODE_TO_TEAM_CLS = {0: (0, "player"), 1: (1, "player"), 2: (0, "goalkeeper"), 3: (1, "goalkeeper"),
                    4: (-1, "referee"), -1: (-1, None)}


class KitClassifier:
    """Team + role from the players the user clicked in the setup UI (``match_setup.json`` -> kits).

    **Nearest clicked samples, per class**: the distance of a crop to a class is the mean distance to its 2 closest
    clicked samples of that class (1 for a class with a single sample). The clicked samples are kept as they are —
    they already cover the lighting the user saw (sun / shade, both halves), so nothing can drift away from them
    (the previous seeded k-means could: on J3 the Sanguedo centre moved further than the distance between the two
    kits). Goalkeepers and referee are classes like the others.

    * unknown (``-1``): farther than ``thr`` from every class (staff, crowd, heavily occluded players);
    * ``margin`` in [0, 1]: 0 = equally close to two classes, 1 = clear.

    :attr:`report` explains the decision rule, including the leave-one-out accuracy on the clicked samples
    themselves (each sample classified by the others) — an honest estimate of how separable the kits are.
    """

    def __init__(self, samples: dict, outlier_quantile: float = 0.97, outlier_factor: float = 1.6, k: int = 2,
                 prune: bool = True):
        self.samples = {c: np.asarray(v, float).reshape(len(v), -1) for c, v in samples.items() if len(v)}
        if "team0" not in self.samples or "team1" not in self.samples:
            raise ValueError("KitClassifier needs at least one clicked sample of each team's outfield kit")
        self.classes = [c for c in KIT_CLASSES if c in self.samples]
        self.q, self.f, self.k = outlier_quantile, outlier_factor, k
        self.thr = np.inf
        self.report: dict = {}
        self.pruned = self._prune_conflicts() if prune else []
        self._fit_from_samples()
        self.report["ignored_samples"] = self.pruned

    def _prune_conflicts(self) -> list:
        """Ignore clicked OUTFIELD samples that the other clicks put in the other team (leave-one-out): on the Canedo
        1080p setup two 'UD Mansores' clicks were mostly grass in the shade and pulled every yellow Canedo shirt in the
        shade to Mansores. Goalkeeper / referee samples are never dropped (often a single click)."""
        out = []
        for c, other in (("team0", "team1"), ("team1", "team0")):
            S = self.samples[c]
            if len(S) < 4:
                continue
            bad = []
            for i in range(len(S)):
                d = self._dist(S[i:i + 1], exclude=(c, i))[0]
                if self.classes[int(np.argmin(d))] == other:
                    bad.append(i)
            if bad and len(S) - len(bad) >= 3:
                out += [(c, i) for i in bad]
        for c in ("team0", "team1"):
            idx = [i for cc, i in out if cc == c]
            if idx:
                self.samples[c] = np.delete(self.samples[c], idx, axis=0)
        return [f"{c}#{i}" for c, i in out]

    # ---- distances
    def _dist(self, X: np.ndarray, exclude: Optional[tuple] = None) -> np.ndarray:
        """(n, n_classes) mean distance to the k nearest samples of each class."""
        X = np.atleast_2d(np.asarray(X, float))
        if len(X) > 5000:                                       # a full match has ~500k crops: bounded memory
            return np.vstack([self._dist(X[i:i + 5000], exclude) for i in range(0, len(X), 5000)])
        out = np.full((len(X), len(self.classes)), np.inf)
        for j, c in enumerate(self.classes):
            S = self.samples[c]
            D = np.linalg.norm(X[:, None] - S[None], axis=2)
            if exclude is not None and exclude[0] == c:
                D[:, exclude[1]] = np.inf
            D.sort(1)
            kk = min(self.k, int(np.isfinite(D[0]).sum()))
            if kk:
                out[:, j] = D[:, :kk].mean(1)
        return out

    def _fit_from_samples(self):
        """Leave-one-out on the clicked samples -> accuracy + a first outlier threshold."""
        ok, n, nn = 0, 0, []
        conf = []
        for c in self.classes:
            for i in range(len(self.samples[c])):
                d = self._dist(self.samples[c][i:i + 1], exclude=(c, i))[0]
                if not np.isfinite(d).any():
                    continue
                pred = self.classes[int(np.argmin(d))]
                n += 1
                ok += pred == c
                if pred != c:
                    conf.append(f"{c}->{pred}")
                j = self.classes.index(c)
                if np.isfinite(d[j]):
                    nn.append(d[j])
        spread = float(np.quantile(nn, 0.9)) if nn else 1.0
        self.thr = 3.0 * spread
        self.report = dict(method="nearest clicked samples per class (k=%d)" % self.k,
                           n_samples={c: int(len(v)) for c, v in self.samples.items()},
                           loo_accuracy=round(ok / n, 3) if n else None, loo_n=n, loo_confusions=conf,
                           outlier_threshold=self.thr)

    def fit(self, feats: Optional[np.ndarray] = None, **_) -> "KitClassifier":
        """Optional: set the unknown-threshold from the match crops themselves (quantile x factor of the distance of
        each crop to its closest class), never below the spread of the clicked samples."""
        if feats is not None and len(feats) >= 20:
            X = np.asarray(feats, float)
            if len(X) > 20000:
                X = X[np.random.default_rng(0).choice(len(X), 20000, replace=False)]
            d = self._dist(X).min(1)
            self.thr = max(self.thr, float(np.quantile(d, self.q) * self.f))
            self.report.update(outlier_threshold=self.thr, n_crops_fit=int(len(X)))
        return self

    def predict(self, feats: np.ndarray, return_margin: bool = False):
        """Codes: 0/1 outfield team, 2/3 goalkeeper of team 0/1, 4 referee, -1 unknown."""
        X = np.asarray(feats, float)
        if len(X) == 0:
            return (np.zeros(0, int), np.zeros(0)) if return_margin else np.zeros(0, int)
        D = self._dist(X)
        j = D.argmin(1)
        srt = np.sort(D, 1)
        d1 = srt[:, 0]
        d2 = srt[:, 1] if D.shape[1] > 1 else np.full(len(X), np.inf)
        margin = np.where(np.isfinite(d2), (d2 - d1) / (d2 + 1e-9), 1.0)
        codes = np.array([CODE[self.classes[i]] for i in j])
        codes = np.where(d1 > self.thr, -1, codes)
        self.report["share_unknown_last_predict"] = float((codes == -1).mean())
        return (codes, margin) if return_margin else codes


def snap_box(box, det_boxes: np.ndarray) -> Optional[np.ndarray]:
    """A box the user dragged (often loose: a lot of grass, sometimes two players) -> the detector box of that player.

    Picks the detection with the highest IoU among those mostly inside the dragged box (>= 60 % of the detection),
    overlapping it well (IoU >= 0.3) or containing it (a small box drawn on the player's chest). ``None`` if nothing matches (keep the dragged box only if it looks like a
    person)."""
    b = np.asarray(box, float)
    D = np.asarray(det_boxes, float).reshape(-1, 4)
    if not len(D):
        return None
    x0 = np.maximum(D[:, 0], b[0]); y0 = np.maximum(D[:, 1], b[1])
    x1 = np.minimum(D[:, 2], b[2]); y1 = np.minimum(D[:, 3], b[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ad = (D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])
    ab = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    iou = inter / (ad + ab - inter + 1e-9)
    ok = (inter / np.maximum(ad, 1e-9) >= 0.6) | (iou >= 0.3) | (inter / ab >= 0.8)     # ... or a small box on him
    if not ok.any():
        return None
    return D[int(np.argmax(np.where(ok, iou, -1)))]


def person_like(box) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return w >= 6 and h >= 14 and 1.1 <= h / max(w, 1e-9) <= 5.0


CODE_NAME = {0: "team0", 1: "team1", 2: "gk0", 3: "gk1", 4: "referee", -1: None}


class TrackVotes:
    """Running per-track votes of the crop codes (live view and final assignment use the same rule)."""

    def __init__(self):
        self.votes: dict = {}
        self.manual: dict = {}            # track id -> code, from a clicked setup sample matched on its frame

    def add(self, tids, codes):
        for tid, c in zip(tids, codes):
            if tid < 0:
                continue
            v = self.votes.setdefault(int(tid), np.zeros(6, int))
            v[int(c) + 1] += 1

    def label(self, tid: int, fallback: int = -1) -> int:
        if tid in self.manual:
            return self.manual[tid]
        v = self.votes.get(int(tid))
        if v is None or v[1:].sum() == 0:
            return fallback
        return int(np.argmax(v[1:]))


def apply_manual(assignment, manual: dict, min_conflict_share: float = 0.8, min_crops: int = 10):
    """Tracks the user clicked in the setup take the clicked class, unless the crop votes of the whole track
    strongly disagree (>= 80 % of >= 10 crops for another class: probably an identity switch of the tracker)."""
    import pandas as pd
    if not manual:
        return assignment
    a = assignment.copy() if assignment is not None and len(assignment) else pd.DataFrame(
        columns=["track_id", "n_crops", "team", "role", "share", "decision"])
    a["clicked"] = ""
    names = {0: "votes_team0", 1: "votes_team1", 2: "votes_gk0", 3: "votes_gk1", 4: "votes_referee"}
    for tid, code in manual.items():
        team, role = CODE_TO_TEAM_CLS[int(code)]
        row = a.index[a.track_id == tid]
        if len(row):
            r = a.loc[row[0]]
            n = int(r.get("n_crops", 0) or 0)
            other = max([int(r.get(v, 0) or 0) for c, v in names.items() if c != code] + [0])
            if n >= min_crops and other / max(n, 1) >= min_conflict_share:
                a.loc[row[0], "clicked"] = CODE_NAME[int(code)]
                a.loc[row[0], "decision"] = "clicked, but the crop votes disagree -> votes kept"
                continue
            a.loc[row[0], ["team", "role", "decision", "clicked"]] = [team, role, "clicked in the setup",
                                                                      CODE_NAME[int(code)]]
        else:
            a = pd.concat([a, pd.DataFrame([dict(track_id=tid, n_crops=0, team=team, role=role, share=1.0,
                                                 decision="clicked in the setup", clicked=CODE_NAME[int(code)])])],
                          ignore_index=True)
    return a


def vote_roles(track_ids: np.ndarray, codes: np.ndarray, min_share: float = 0.6) -> "pd.DataFrame":
    """Per track: votes for each code, winner and its share (for explainability -> team_assignment.csv)."""
    import pandas as pd
    df = pd.DataFrame({"track_id": track_ids, "code": codes})
    df = df[df.track_id >= 0]
    if len(df) == 0:
        return pd.DataFrame(columns=["track_id", "n_crops", "team", "role", "share", "decision"])
    ct = df.groupby(["track_id", "code"]).size().unstack(fill_value=0)
    names = {0: "votes_team0", 1: "votes_team1", 2: "votes_gk0", 3: "votes_gk1", 4: "votes_referee", -1: "votes_unknown"}
    ct = ct.rename(columns=names)
    for c in names.values():
        if c not in ct:
            ct[c] = 0
    ct = ct[list(names.values())]
    n = ct.sum(1)
    known = ct.drop(columns="votes_unknown")
    best = known.idxmax(1)
    share = known.max(1) / n.clip(lower=1)
    inv = {v: k for k, v in names.items()}
    tie = known.eq(known.max(1), axis=0).sum(1) > 1          # two classes with the same top count -> undecided
    code = best.map(inv).where((known.max(1) > 0) & ~tie, -1)
    out = ct.copy()
    out["n_crops"] = n
    out["team"] = [CODE_TO_TEAM_CLS[int(c)][0] for c in code]
    out["role"] = [CODE_TO_TEAM_CLS[int(c)][1] or "unknown" for c in code]
    out["share"] = share.round(3)
    out["decision"] = np.where(code < 0, np.where(tie & (known.max(1) > 0), "tie -> unknown", "unknown"),
                               np.where(share >= min_share, "majority", "plurality (ambiguous)"))
    return out.reset_index()
