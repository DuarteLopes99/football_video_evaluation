"""football_eval — single-camera football match analysis.

Two layers, deliberately decoupled:

* **Perception** (video -> tracking table in pitch metres)
  ``video`` -> ``detection`` -> ``tracking`` -> ``teams`` -> ``pitch`` (homography) -> ``ball``
  orchestrated by :mod:`football_eval.perception`.

* **Analytics** (tracking table -> possession, events, xG, stats, timeline, report)
  :mod:`possession`, :mod:`events`, :mod:`xg`, :mod:`stats`, :mod:`timeline`, :mod:`report`
  orchestrated by :mod:`football_eval.analytics`.

The analytics layer only needs a tracking table, so it can be developed and evaluated on
synthetic data (:mod:`football_eval.synthetic`) or on public tracking data while the perception
models are still being fine-tuned. See ``docs/STATS_RULEBOOK.md`` for every rule and threshold.
"""
__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .pitch import Pitch  # noqa: F401
