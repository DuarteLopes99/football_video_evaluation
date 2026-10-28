"""Configuration: nested dataclasses with defaults, overridable from a YAML file.

Every numeric threshold used by the analytics rules lives here so that the rulebook
(docs/STATS_RULEBOOK.md) can reference it by key, e.g. ``events.shot_min_speed``.
"""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import Any, Optional

import yaml


@dc.dataclass
class VideoCfg:
    analysis_fps: float = 10.0          # frames per second actually processed (stride = fps_src / analysis_fps)
    start_s: float = 0.0
    end_s: Optional[float] = None       # None = until the end of the file
    resize_width: Optional[int] = None  # downscale before detection (None = native)
    only_periods: bool = True           # with periods set in the setup: skip pre-match, half-time and post-match
    period_margin_s: float = 20.0       # ... keeping this much video before / after each period
    ignore_regions: list = dc.field(default_factory=list)  # [[x0,y0,x1,y1], ...] overlays (scoreboard, logos)


@dc.dataclass
class DetectionCfg:
    backend: str = "yolox_onnx"         # "yolox_onnx" (cv2.dnn, no torch) | "ultralytics"
    model_path: str = "models/yolox.onnx"
    # map model class names -> canonical classes {player, goalkeeper, referee, ball}
    class_map: dict = dc.field(default_factory=lambda: {
        "person": "player", "sports ball": "ball",
        "player": "player", "goalkeeper": "goalkeeper", "referee": "referee", "ball": "ball"})
    conf_player: float = 0.30
    conf_ball: float = 0.15
    imgsz: int = 1280                   # ultralytics inference size
    tile: int = 1024                    # yolox_onnx: max tile size (px). Smaller = more tiles = slower, better ball
    tile_overlap: int = 96              # yolox_onnx: overlap between tiles (px); people cut at a seam are merged
    tile_stride: Optional[int] = None   # deprecated (old fixed grid), ignored
    pitch_roi: bool = True              # only look at the calibrated pitch (+ margin): fewer tiles, no crowd boxes
    runtime: str = "auto"               # yolox_onnx: "auto" (onnxruntime, CoreML if faster, else cv2.dnn) | "cv2" | "coreml"
    threads: int = 0                    # onnxruntime threads (0 = all cores)
    device: str = "cpu"                 # ultralytics: "cpu" | "mps" | "cuda:0"
    min_box_h_frac: float = 0.012       # drop boxes smaller than this fraction of image height


@dc.dataclass
class TrackingCfg:
    backend: str = "iou"                # "iou" (built-in Hungarian) | "bytetrack" (supervision)
    max_age_s: float = 1.0
    min_iou: float = 0.2
    max_center_dist_px: float = 80.0


@dc.dataclass
class TeamsCfg:
    backend: str = "color"              # "color" (HSV/Lab torso descriptor) | "siglip" (transformers)
    fit_frames: int = 60                # sampled frames used to fit the 2-team clustering
    outlier_quantile: float = 0.97      # crops farther than this quantile x factor => referee/other
    outlier_factor: float = 1.6
    team_names: list = dc.field(default_factory=lambda: ["Team A", "Team B"])
    kit_samples: Optional[dict] = None  # {team0|team1|gk0|gk1|referee: [{t, box}, ...]} clicked in the setup UI
                                        # -> seeded (supervised) team / role assignment, see teams.KitClassifier
    kit_colors: Optional[list] = None   # shirt colours in team_names order, BGR e.g. [[0, 215, 255], [40, 90, 30]]
                                        # -> clusters are mapped to names by colour (else cluster order is arbitrary)


@dc.dataclass
class PitchCfg:
    length: float = 105.0
    width: float = 68.0
    homography: str = "static"          # "static" | "keypoints" | "flow" | "ptz" (fixed-position pan/tilt/zoom camera)
    calibration_file: Optional[str] = None   # json with image<->pitch correspondences or a camera model
    keypoint_model: Optional[str] = None     # ultralytics pose model trained on pitch keypoints
    keypoint_layout: str = "roboflow32"
    keypoint_conf: float = 0.5
    offpitch_margin_m: float = 2.5      # people further outside the lines are bench / staff / crowd
    ptz_refine_every: int = 1           # ptz: align the pitch template to the white lines every N frames
    ptz_max_err_px: float = 4.0         # ptz: mean line error (px) above which the tracker tries to relocalise
    reanchor_every_s: float = 0.0       # flow mode: force re-anchoring with keypoint model (0 = never)
    fixed_size: bool = False            # True = length/width were measured by the user: they override the size
                                        # stored in a PTZ venue file (which then needs re-solving if they differ)
    calibration_points: Optional[dict] = None   # static mode: {landmark: [u, v]} clicked in the setup UI
    flow_anchor_points: Optional[dict] = None   # flow mode: {frame_idx: {landmark: [u, v]}} from the setup UI
    extra_anchors: Optional[dict] = None        # ptz mode: {frame_idx: [pan, tilt, f]} poses validated in the UI


@dc.dataclass
class BallCfg:
    detector: str = "model"             # "model" (detector's ball class) | "motion" (low-res streams) | "both"
    max_speed_ms: float = 40.0          # physically impossible jumps are discarded
    max_gap_s: float = 1.0              # interpolate gaps up to this length
    smooth_window: int = 5              # Savitzky-Golay window (frames at analysis fps)


@dc.dataclass
class PossessionCfg:
    control_radius_m: float = 1.5       # ball within this distance of a player's feet => candidate control
    duel_radius_m: float = 1.5          # >=1 player of each team within this radius => duel
    max_ball_speed_control: float = 9.0  # a ball faster than this is in flight, not controlled
    min_control_s: float = 0.2          # minimum duration of a touch (one analysed frame counts 1/fps)
    team_switch_s: float = 0.6          # hysteresis for team possession changes


@dc.dataclass
class EventsCfg:
    max_pass_s: float = 6.0
    min_pass_length_m: float = 3.0      # shorter ball travel between two controllers is a contest, not a pass
    progressive_pass_m: float = 10.0
    shot_min_speed: float = 11.0        # m/s right after release
    shot_max_dist: float = 35.0         # m from goal centre
    shot_goal_tolerance: float = 3.0    # m outside the posts the ball line may cross the goal line
    block_window_s: float = 0.8
    block_max_dist: float = 12.0
    dead_ball_min_s: float = 3.0        # ball unseen / out for at least this long => stoppage
    out_margin_m: float = 0.5           # ball beyond the line by this margin => out of play
    still_speed_ms: float = 1.6         # mean player speed below this during a stoppage => play stopped
    static_ball_spread_m: float = 0.8   # ball position spread (2 s window) below this => ball at rest
    recovery_min_s: float = 0.3         # new controller must keep the ball this long to count a recovery
    corner_radius_m: float = 2.5
    throwin_band_m: float = 2.0
    near_edge_band_m: float = 3.0       # near touchline out of view: ball last seen this close to the bottom edge
    kickoff_radius_m: float = 3.0
    penalty_radius_m: float = 2.0
    goal_confirm_window_s: float = 150.0  # kick-off must follow a goal within this time
    goal_min_stoppage_s: float = 20.0   # celebrations: a goal stoppage lasts at least this long
    kickoff_own_half_share: float = 0.9  # share of each team's players in own half at a kick-off
    kickoff_half_margin_m: float = 1.5  # tolerance on the halfway line for that check (position noise)
    static_still_min_s: float = 5.0     # ball at rest AND players still this long => dead ball (foul, injury)
    static_ball_min_s: float = 6.0      # ball at rest this long => dead ball even if players move
    final_third_x: float = 70.0         # attacking third starts here (attacking left->right)
    ppda_zone_x: float = 42.0           # PPDA zone: our defensive actions beyond this x (opp. passes before L - x)


@dc.dataclass
class MatchCfg:
    home_attacks_right_first_half: Optional[bool] = None  # None = infer from average positions
    halves: Optional[list] = None       # [[start_s, end_s], [start_s, end_s]] of the video; None = infer
    half_length_min: float = 45.0


@dc.dataclass
class ScoreboardCfg:
    enabled: bool = False
    roi: Optional[list] = None          # [x0,y0,x1,y1] of the score digits on the broadcast overlay
    change_threshold: float = 18.0      # mean abs diff (0-255) to flag a change
    min_persist_s: float = 3.0


@dc.dataclass
class XGCfg:
    model_file: str = "models/xg_model.json"
    variant: str = "full"               # "basic" (distance+angle) | "full" (adds freeze-frame features)


@dc.dataclass
class OpportunitiesCfg:
    """Major attacking opportunities (football_eval/opportunities.py, rulebook §7c)."""
    big_chance_xg: float = 0.30         # best shot xG >= this => "Big chance" (same threshold as the big-chances stat)
    flag_min_xg: float = 0.10           # any shot at least this good is flagged, whatever its outcome
    merge_window_s: float = 6.0         # same-team components closer than this are one opportunity
    include_box_entries: bool = True    # ball reaches the opponent box (pass or carry) without a shot
    include_set_pieces: bool = True     # corners and free kicks near goal
    set_piece_max_dist: float = 30.0    # free kick within this distance of the goal centre (m)
    carry_min_m: float = 3.0            # a carry into the box must move the ball at least this far
    corner_xg_prior: float = 0.03       # chance value of a corner without a shot (league average ~0.02-0.04)
    free_kick_xg_prior: float = 0.05    # used when no xG model is loaded
    no_shot_discount: float = 0.5       # danger of a box entry = discount x xG of a shot from that spot
    clip_before_s: float = 10.0         # clip starts this long before the first moment (build-up)
    clip_after_s: float = 6.0           # ... and ends this long after the last one
    auto_clips: bool = True             # cut clips of flagged opportunities after each run (needs the video)


@dc.dataclass
class Config:
    video: VideoCfg = dc.field(default_factory=VideoCfg)
    detection: DetectionCfg = dc.field(default_factory=DetectionCfg)
    tracking: TrackingCfg = dc.field(default_factory=TrackingCfg)
    teams: TeamsCfg = dc.field(default_factory=TeamsCfg)
    pitch: PitchCfg = dc.field(default_factory=PitchCfg)
    ball: BallCfg = dc.field(default_factory=BallCfg)
    possession: PossessionCfg = dc.field(default_factory=PossessionCfg)
    events: EventsCfg = dc.field(default_factory=EventsCfg)
    match: MatchCfg = dc.field(default_factory=MatchCfg)
    scoreboard: ScoreboardCfg = dc.field(default_factory=ScoreboardCfg)
    xg: XGCfg = dc.field(default_factory=XGCfg)
    opportunities: OpportunitiesCfg = dc.field(default_factory=OpportunitiesCfg)
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        return dc.asdict(self)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(p: Optional[str]) -> Optional[str]:
    """Paths in configs may be relative to the working directory or to the project root."""
    if not p:
        return p
    q = Path(p).expanduser()
    if q.exists():
        return str(q)
    if q.is_absolute() and q.suffix.lower() not in (".mp4", ".mov", ".m4v", ".mkv", ".avi"):
        return str(q)
    r = PROJECT_ROOT / q
    if r.exists():
        return str(r)
    if q.suffix.lower() in (".mp4", ".mov", ".m4v", ".mkv", ".avi"):   # a recording moved to a video library
        from .storage import find_video
        found = find_video(q.name)
        if found:
            return found
    return str(q)


# Processing speed presets (run_analysis.py --speed, launcher "Speed"). The detector is ~95 % of the time:
# analysed frames per second x detector tiles per frame is what you pay for.
SPEED_PRESETS = {
    "fast": dict(video=dict(analysis_fps=5.0), detection=dict(tile=1024, tile_stride=None),
                 note="5 analysed fps: ~2x faster than standard; short passes and fast shots are missed more often"),
    "standard": dict(video=dict(analysis_fps=10.0), detection=dict(tile=1024, tile_stride=None),
                     note="10 analysed fps, 2 tiles on a 1080p pitch"),
    "quality": dict(video=dict(analysis_fps=10.0), detection=dict(tile=800, tile_stride=None),
                    note="10 analysed fps, smaller tiles (3-6): more distant players and balls, ~2-3x slower than standard"),
}


def apply_speed(cfg: "Config", preset: Optional[str]) -> list:
    """Apply a speed preset on top of the config (the preset wins). Returns a change log."""
    if not preset:
        return []
    if preset not in SPEED_PRESETS:
        raise ValueError(f"unknown speed preset {preset}: {', '.join(SPEED_PRESETS)}")
    log = []
    for sec, vals in SPEED_PRESETS[preset].items():
        if sec == "note":
            continue
        for k, v in vals.items():
            old = getattr(getattr(cfg, sec), k)
            if old != v:
                setattr(getattr(cfg, sec), k, v)
                log.append(f"{sec}.{k}: {old} -> {v}")
    return log


def _merge(obj: Any, data: dict) -> Any:
    for k, v in (data or {}).items():
        if not hasattr(obj, k):
            raise KeyError(f"Unknown config key: {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dc.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_config(path: Optional[str | Path] = None, **overrides) -> Config:
    """Defaults <- YAML file <- keyword overrides (``section={...}``)."""
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    _merge(cfg, overrides)
    return cfg
