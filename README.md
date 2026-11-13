# Football video evaluation

Football analytics from one camera, in two independent projects that share one dashboard and one Python
environment:

| Folder | What it does | Docs |
|---|---|---|
| [`global_video_analysis/`](global_video_analysis/) | Full-match analysis of an mp4: detection, tracking, teams, camera, ball → events, xG, team / player stats, timeline, flagged opportunities, report | [`README`](global_video_analysis/README.md), [`USER_GUIDE`](global_video_analysis/docs/USER_GUIDE.md) |
| [`offside_code/`](offside_code/) | Offside / onside on a single frame: semi-manual camera calibration from a set of images, then the decision with geometrically correct offside lines and uncertainty | [`README`](offside_code/README.md), [`STUDIO_GUIDE`](offside_code/docs/STUDIO_GUIDE.md) |

## Start

Double-click **`start.command`** (macOS; `./start.command` on Linux, **`start.bat`** on Windows) — or the
*Football Video Analysis* shortcut on the Desktop. The first
run creates `.venv/` and installs both projects' requirements; then the dashboard opens at
http://127.0.0.1:8770:

* tabs **Analyse · Manual decisions · Match screen · Results & opportunities** — `global_video_analysis`;
* tab **Offside** — `offside_code` (venues, click-to-mark calibration, decision on a chosen frame);
* tabs **Jobs · Guide** — background jobs and every guide (including the Offside tab guide).

```bash
./start.command                                   # dashboard
./start.command --open offside                    # straight to the Offside tab
./start.command --video <file.mp4> --start        # process a recording at once
```

From a terminal with the venv active, each project also runs on its own:
`python global_video_analysis/scripts/launcher.py`, `python offside_code/offside_studio.py`, and all the CLIs
described in the two READMEs (run them from inside their folder).

## Windows

Supported the same way as macOS / Linux. Install Python 3.10+ from python.org (tick *Add python.exe to PATH*), then
double-click **`start.bat`**: it creates `.venv\`, installs the packages, switches to the GPU build on an NVIDIA
computer and opens the dashboard. Optional: [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) on the PATH cuts the
opportunity clips faster (without it OpenCV re-encodes them). Notes:

* everything is read and written as UTF-8 (`start.bat` also sets `PYTHONUTF8=1`), so setups, reports and logs made
  on the Mac open unchanged on Windows and back; relative paths in the setups always use `/`;
* the PC is kept awake while a match is processed (like `caffeinate` on the Mac);
* *Desktop shortcut* writes a `.bat` (also into a OneDrive Desktop) that goes through `start.bat`;
* `cut_highlights.sh` / `cut_opportunities.sh` in the outputs are bash scripts; on Windows use
  `python scripts/cut_opportunities.py --run outputs/<match>` or the *Cut clips* button instead;
* video libraries in `configs/storage.yaml` that point to Mac folders are skipped; add the Windows folders there
  (recordings are also found by file name, so a setup made on the Mac finds its video on the PC).

From a terminal: `.venv\Scripts\activate`, then the same commands as below with `python`.

## GPU (automatic)

The analysis runs on any laptop's CPU and uses a GPU by itself when the computer has one: the person / ball detector
(~95 % of the processing time) benchmarks every accelerator it can reach and keeps the fastest.

| Computer | What is used | Setup |
|---|---|---|
| NVIDIA GPU (Windows / Linux) | onnxruntime CUDA (TensorRT with `--device tensorrt`); PyTorch models on `cuda:0` | none: `start.bat` / `start.command` install `onnxruntime-gpu[cuda,cudnn]` once when `nvidia-smi` works (needs a recent NVIDIA driver) |
| Other GPU on Windows | DirectML | `pip uninstall onnxruntime && pip install onnxruntime-directml` |
| Mac (Apple Silicon) | CoreML when faster than the CPU; PyTorch models on `mps` | none |
| No GPU | CPU (all cores) | none |

Check a computer: `cd global_video_analysis && python -m football_eval.accel`. Force the CPU: `run_analysis.py
--device cpu`, or `detection.runtime: cpu` in the config. `offside_code` uses OpenCV's CUDA backend when OpenCV was
built with it (`OFFSIDE_DEVICE=cpu` to disable); it works on one frame, so the CPU is enough there.

## Layout

```
football_video_evaluation/
├── start.command            dashboard launcher, macOS / Linux (shared .venv, installs the GPU build when there is one)
├── start.bat                same on Windows
├── .venv/                   Python environment of both projects (git-ignored)
├── global_video_analysis/   match analysis  (football_eval/, scripts/, configs/, docs/, notebooks/, tests/, outputs/, data/)
└── offside_code/            single-frame offside (offside/, configs/, calibration/, venues/, docs/, notebooks/, tests/)
```

The two projects do not import each other, except that the dashboard (`global_video_analysis/football_eval/ui/
launcher.py`) mounts `offside_code/offside/studio.py` under `/off/`, and `global_video_analysis` can re-use the
YOLOX model and a stationary camera model of `offside_code` (`scripts/get_models.py`, `configs/example_anadia.yaml`).
