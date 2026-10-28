# Video storage — where the match recordings live

## 1 · The folder

All recordings go in **`data/videos/`** (git-ignored). Drop a file on the launcher or copy it there; name it
`YYYY-MM-DD_<Home>_vs_<Away>_<round>.mp4` so the list sorts by date. The Canedo match was moved there from
`data/raw/`. Older setups, runs and notebooks that still say `data/raw/...` keep working, because recordings are
**found by file name** in every library (`football_eval/storage.py → find_video`).

`data/videos` is the first **video library** of `configs/storage.yaml`. A library is just a folder. You can add
more of them (an external drive, a cloud folder), and the launcher lists the videos of all of them.

## 2 · How much space it takes

| Recording | Bitrate | 2-h file | 30-match season |
|---|---|---|---|
| Club stream as today (CFC TV, 640 × 360, 30 fps) | ~320 kb/s | **~0.3 GB** (Canedo: 289 MB) | **~9 GB** |
| Same match at 720p | ~1.5–2 Mb/s | ~1.5 GB | ~45 GB |
| Same match at 1080p (own camera / phone) | ~3–4 Mb/s | ~3 GB | ~100 GB |

What an analysed match keeps in `outputs/<match>/`: tracking tables + report (~12 MB on the Canedo segment, more for
a full match), plus the clips of the flagged opportunities (~0.5–1 MB each at 640 × 360). That is enough to re-run
the analytics, re-open the report and watch every chance **without the full video**.

When the full video is needed:

| Step | Needs the full video locally? |
|---|---|
| Setup UI, processing (`run_analysis.py --video`), camera repair, debug video | **yes** (read from start to end / random access) |
| Review UI | yes (it shows frames) |
| Re-running the analytics (`--tracks`), report, opportunities list | no |
| Watching the flagged opportunities | no — the clips are in `outputs/<match>/clips/opportunities/` |
| "full video @ mm:ss" links in the report / launcher player | yes, or wherever the archive library is reachable |

## 3 · Options so the computer does not have to hold every match

**A · Cloud folder with on-demand files (recommended).** Google Drive for desktop in *Stream files* mode, OneDrive
*Files On-Demand*, iCloud Drive with *Optimise Mac Storage*, or Dropbox *online-only*. These put a folder on your Mac
that shows every file while only downloading the ones you open. Add it as a library and set it as the archive:

```yaml
libraries:
  - name: Local
    path: data/videos
  - name: Archive
    path: ~/Library/CloudStorage/GoogleDrive-<your-account>/My Drive/Football/matches
archive: Archive
```

Workflow: record → drop in **Local** → setup, process, review → **Move to archive** (launcher button). The cloud
client uploads it. Then free the local copy (Finder: right-click → *Remove download* / *Free up space*). If you need
it again (re-process with a better model, review again), it is downloaded the moment the pipeline opens it. The
launcher marks such files "in the cloud". Check your plan's storage before archiving a whole season. At today's
640 × 360, a season fits in ~9 GB.

**B · External drive.** Add `/Volumes/<drive>/matches` as a library. It is the cheapest per GB and the fastest to
process from (USB-C SSD). The launcher shows the library only while the drive is connected.

**C · Leave the full match where it was published.** If the club stream stays online, keep only the outputs and the
clips. When you need the full video again, download it to `data/videos`, re-process, and delete it. (Only for
recordings you are allowed to download.)

**D · Keep only the clips.** After the review, the flagged clips plus `outputs/` are the useful part of a match
(~20–40 MB). Deleting the full video is final, so archive it first (A/B) if you might re-process it later.

Not recommended: re-encoding the 320 kb/s streams to save space. They are already heavily compressed, and every
re-encode costs detail on a 2–5 px ball.

## 4 · Settings (`configs/storage.yaml`)

| Key | Meaning |
|---|---|
| `libraries` | list of `{name, path, note}`. Paths: relative to the project, absolute, or `~/…`. The **first** library receives the videos dropped on the launcher. |
| `archive` | name of the library the **Move to archive** button moves an analysed recording to (`null` hides the button) |

Code: `football_eval/storage.py` (listing, cloud-only detection, lookup by name, move). `config.resolve_path` uses
the lookup for any video path that no longer exists.
