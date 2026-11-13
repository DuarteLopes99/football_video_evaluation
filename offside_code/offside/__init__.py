"""offside — single/multi-frame offside lines from broadcast football frames."""
from .camera import project, ground, rays, vanishing_point, make_camera, homography
from .template import all_element_names, polyline
from .features import (line_mask, extract_features, find_candidates, draw_candidates,
                       features_from_candidates, draw_features)
from .calibration import Frame, Priors, calibrate
from .players import PlayerDetector, assign_teams
from .decision import decide, monte_carlo, summarise, lift_player, offside_line
from .render import render_scenario, render_birds_eye, draw_pitch_model
from .pipeline import run, load_frame

__version__ = '0.2.0'
