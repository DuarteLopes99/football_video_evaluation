"""'Extra person' kit classes (coach / fan / police / security) and the model files that are not in git.

    python -m pytest tests/test_extra_person_models.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_eval import models  # noqa: E402
from football_eval.live import NOT_PLAYING, label_texts, row_label  # noqa: E402
from football_eval.match_setup import KIT_CLASSES as SETUP_CLASSES, apply_to_config, checklist, new_setup  # noqa: E402
from football_eval.people import OFF_PITCH_TYPES, TYPE_LABEL, people_summary  # noqa: E402
from football_eval.teams import (CODE, CODE_LABEL, CODE_TO_TEAM_CLS, EXTRA_CODES, KIT_CLASSES, KitClassifier,  # noqa: E402
                                 NOT_PLAYING_ROLES, TrackVotes, apply_manual, vote_roles)


def blobs(center, n, rng, spread=0.05):
    return list(np.asarray(center, float) + rng.normal(0, spread, (n, len(center))))


def test_classes_consistent():
    assert tuple(SETUP_CLASSES) == tuple(KIT_CLASSES)
    assert {CODE_TO_TEAM_CLS[c][1] for c in EXTRA_CODES} == set(NOT_PLAYING_ROLES) == {"staff", "spectator", "police", "steward"}
    assert all(CODE_TO_TEAM_CLS[c][0] == -1 for c in EXTRA_CODES)              # never a team
    assert set(NOT_PLAYING_ROLES) <= set(OFF_PITCH_TYPES) and set(NOT_PLAYING_ROLES) <= set(NOT_PLAYING)
    assert TYPE_LABEL["police"] == "POL" and label_texts(None)["police"] == "POL"
    assert CODE_LABEL[CODE["police"]] == "police" and CODE_LABEL[CODE["coach"]] == "staff"
    assert row_label("police", -1) == "police"


def test_classifier_learns_extra_people():
    rng = np.random.default_rng(0)
    dim = 6
    c0, c1, cp, cc = np.eye(dim)[0], np.eye(dim)[1], np.eye(dim)[2], np.eye(dim)[3]
    kit = KitClassifier({"team0": blobs(c0, 5, rng), "team1": blobs(c1, 5, rng),
                         "police": blobs(cp, 3, rng), "coach": blobs(cc, 2, rng)})
    assert kit.has_extra
    pred = kit.predict(np.vstack([blobs(c0, 3, rng), blobs(cp, 3, rng), blobs(cc, 3, rng)]))
    assert list(pred[:3]) == [0] * 3 and list(pred[3:6]) == [CODE["police"]] * 3 and list(pred[6:]) == [CODE["coach"]] * 3
    assert kit.report["loo_accuracy"] == 1.0
    assert not KitClassifier({"team0": blobs(c0, 3, rng), "team1": blobs(c1, 3, rng)}).has_extra


def test_votes_and_manual_clicks_give_not_playing_roles():
    tids = np.array([1, 1, 1, 2, 2, 2])
    codes = np.array([CODE["police"]] * 3 + [0, 0, CODE["fan"]])
    a = vote_roles(tids, codes)
    r = a.set_index("track_id")
    assert r.loc[1, "role"] == "police" and r.loc[1, "team"] == -1 and r.loc[1, "votes_police"] == 3
    assert r.loc[2, "role"] == "player" and r.loc[2, "team"] == 0
    m = apply_manual(a, {2: CODE["security"], 9: CODE["coach"]}).set_index("track_id")
    assert m.loc[2, "role"] == "steward" and m.loc[2, "team"] == -1 and m.loc[2, "clicked"] == "security"
    assert m.loc[9, "role"] == "staff"
    tv = TrackVotes()
    tv.add(np.array([5, 5]), np.array([CODE["police"], CODE["police"]]))
    assert tv.label(5) == CODE["police"]


def test_people_summary_and_setup():
    tr = pd.DataFrame(dict(frame=[0, 0, 1], track_id=[1, 2, 1], cls=["police", "player", "police"], team=[-1, 0, -1]))
    s = people_summary(tr).set_index("type")
    assert s.loc["police", "tracks"] == 1 and s.loc["player", "tracks"] == 1
    setup = new_setup("x.mp4")
    assert all(c in setup["kits"] for c in ("coach", "fan", "police", "security"))
    setup["kits"]["police"].append(dict(frame=10, t=0.4, box=[1, 2, 3, 4]))
    items = {i["item"]: i for i in checklist(setup, Path("."))}
    assert items["kits: extra people"]["status"] == "ok" and "1 police" in items["kits: extra people"]["detail"]
    from football_eval.config import load_config
    cfg = load_config()
    apply_to_config(setup, cfg)
    assert len(cfg.teams.kit_samples["police"]) == 1


def test_models_fetch_rules(tmp_path, monkeypatch):
    lfs = tmp_path / "yolox.onnx"
    lfs.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 35858002\n")
    assert not models.usable(lfs)                                    # a git-LFS pointer is not a model
    sib = tmp_path / "sibling.onnx"
    sib.write_bytes(b"\0" * 1_100_000)
    monkeypatch.setitem(models.MODELS, "yolox.onnx", dict(models.MODELS["yolox.onnx"], siblings=[sib]))
    got = models.ensure("yolox.onnx", lfs, log=lambda *_: None)      # replaced by the sibling's copy
    assert got == lfs and lfs.stat().st_size == 1_100_000
    with pytest.raises(models.ModelMissing, match="detector model not found"):
        models.resolve_model(str(tmp_path / "my_detector.onnx"))   # a custom missing model: clear error
    assert models.resolve_model(str(lfs)) == str(lfs)


def test_models_download_failure_is_explained(tmp_path, monkeypatch):
    monkeypatch.setitem(models.MODELS, "yolox.onnx", dict(models.MODELS["yolox.onnx"], siblings=[],
                                                          url="http://127.0.0.1:9/none.onnx"))
    with pytest.raises(models.ModelMissing, match="get_models.py"):
        models.ensure("yolox.onnx", tmp_path / "yolox.onnx", log=lambda *_: None)
    assert not (tmp_path / "yolox.onnx.part").exists() and not (tmp_path / "yolox.onnx").exists()
