# Report guide: what's in the report, which numbers matter, and how far to trust them

The footage is one camera that pans and zooms at 640 × 360, with no score overlay and no audio. That camera
sees roughly a third to a half of the pitch at any time. Players are about 25 px tall and the ball 2–5 px.
So **not every number is equally trustworthy**, and the report says so explicitly. Each statistic carries a
**confidence** computed for *that* match, together with the reason. Grading code: `football_eval/confidence.py`.

## 1 · What the report contains, top to bottom

| Section of `report.html` | What it tells you | File |
|---|---|---|
| **Score** | Final score. Marked *inferred* unless it matches your match sheet | — |
| **How to read it** | The four confidence levels (below) | — |
| **⚑ Major opportunities** | Every attacking move with a shot, goal, penalty or box entry. The flagged ones (goals, penalties, big chances, shots on target, shots with xG ≥ 0.10) come first, each with its match clock, video time, chance xG, a ▶ clip and a link to that moment in the full video. Quick links to the goals and the best chances sit above the table. The other moments and a per-team summary are collapsed below. Rules: rulebook §7c | `opportunities.csv`, `clips/opportunities/` |
| **Data quality** | The share of play where the camera calibration was usable, how often the ball was found, whether you checked frames and events, whether the match sheet is consistent, and how many camera stretches you repaired | `data_quality.json` |
| **Key metrics you can rely on** | High and Medium metrics, with the reason for each grade | `key_metrics.csv` |
| **Indicative metrics** | Low-confidence metrics: use them for trends and with care | `key_metrics.csv` |
| **Experimental metrics** (collapsed) | Computed, but not reliable on this footage yet | `key_metrics.csv` |
| **Timeline** | Goals, shots, cards, substitutions and set pieces with video time and match clock. ✓ = confirmed by you or the sheet | `timeline.csv` |
| **Key events and why** | One sentence per goal, shot or card: the rule, the measured values and the thresholds | `events.csv` (`why`) |
| **Players by shirt number** | Only the events you attributed in the review or on the match sheet | `player_stats_by_number.csv` |
| **Figures** | Opportunities over the match (height = danger, filled = flagged), shot map, possession over time, average positions | — |
| **How this report was produced** | Every input and who provided it (manual / measured / auto / default / assistant) | `provenance.json` |
| **Measured reliability** | What your review measured, with 95 % intervals | `validation_summary.csv` |
| **Rejected events** | What the rules found that you, the match sheet or an excluded stretch removed | `events_rejected.csv` |
| **Full table** (collapsed) | Every team statistic, unchanged | `team_stats.csv` |

## 2 · Confidence levels

| Level | Meaning | Use it to… |
|---|---|---|
| **High** | Comes from the match sheet, or was checked event by event | quote it as a fact |
| **Medium** | Right order of magnitude; your review confirmed the input it depends on | compare the two teams, or the two halves |
| **Low** | Biased by the camera (partial view, missed or false ball); not verified | spot trends, with "approximately" |
| **Experimental** | Depends on a signal that is not reliable at this resolution (mostly the ball) | nothing yet; kept to measure progress |

Every grade starts from the recording profile and moves with what was measured on the match. The *why* column
lists each move, for example *"every detected shot was checked by you"* or *"only 50 % of shot locations were
clicked: xG still uses the tracked position"*.

## 3 · Metric by metric: value, and confidence on this kind of recording

★ = headline value for a coach. "Default" = no manual input. "Best case" = full setup and review done.

| Metric | Tells you | Value | Default | Best case | What raises it |
|---|---|---|---|---|---|
| **Goals** | the result | ★ | Low (inferred from kick-offs) | **High** | final score + goal minutes on the match sheet |
| **Cards** | discipline | ★ (red) | — | **High** | match sheet |
| **Shots, shots on target** | volume of chances | ★ | Low | **Medium** | review every detected shot; add the missing ones |
| **xG, big chances** | quality of chances, compared with goals | ★ | Low | **Medium** | the above + click every shot location (and the keeper) |
| **Flagged opportunities ⚑** | how many chances worth watching, each with a clip | ★ | Low | **Medium** | review every detected shot (they make up the list); add the missing ones |
| **Defensive line height** | high press vs deep block | ★ | Low | **Medium** | frame check confirms the lines (≥ 60 %), kit clicks, direction given |
| **Block depth / width** | compactness | ★ | Low | **Medium** | as above |
| **Team centre of mass** | territory from player positions | | Low | **Medium** | as above |
| Possession %, field tilt % | ball dominance / territory with the ball | ★ | Low | Low | needs a better ball: 720p recordings or a fine-tuned ball detector |
| Corners, throw-ins, goal kicks, free kicks | set-piece counts | | Low | Low | same, and review them (planned) |
| Distance covered (in view), average speed | physical comparison of the teams | | Low | Low | only players the camera sees; never compare players |
| Passes, pass accuracy, progressive passes, entries | build-up | | Experimental | Experimental | the ball at 2–5 px misses passes and invents others |
| Interceptions, recoveries, PPDA | pressing and regains | | Experimental | Experimental | same |
| Fouls (proxy), offsides (suspected) | discipline proxies | | Experimental | Experimental | no whistle audio; inferred from restarts |
| Sprints | intensity | | Experimental | Experimental | position noise turns into false sprints |
| *Not in the report* | per-tracklet player stats (`player_stats.csv`), headers vs feet, shot height | | — | — | need higher resolution and player identity |

### On the Canedo recording, with the setup and review done

The numbers worth reading, in order:

1. **The result, the scorers and the cards**: High, from your match sheet.
2. **Shots, shots on target, xG and big chances**: Medium, once every shot is reviewed and located. On this footage
   these are the best summary of who created more.
3. **Team shape** (defensive line height, block depth and width, centre of mass): Medium, from player positions,
   which are reliable when the camera is right. They describe *how each team defended*, which is the camera's
   strongest signal. They are measured only on the part of the pitch in view, which is usually around the ball.
4. Possession and field tilt: Low. Mention a big gap (for example 64–36) only as "clearly more of the ball".
5. Everything about passes and pressing: Experimental. Don't use it until the ball detection improves (720p or a
   trained ball model; see `docs/SOTA_REVIEW.md`).

## 4 · The rules behind the grade

| Family | Start | Down | Up |
|---|---|---|---|
| Goals | Low | — | High if goals = final score on the sheet |
| Shot-based (shots, xG…) | Low | Experimental if under 50 % of the reviewed shots were real and the review is incomplete | Medium if ≥ 90 % of detected shots were reviewed; xG stays Low until ≥ 80 % of shot locations are clicked |
| Position-based (shape, distance) | Medium (shape), Low (physical) | Low if the frame check wasn't done, lines were right on under 60 % of checked frames, camera usable on under 50 % of play, no kit clicks, or (for directional ones) the direction wasn't given | — |
| Ball-based (possession, passes, set pieces) | Low / Experimental | Experimental if the ball was found in under 40 % of frames or the camera was usable on under 50 % of play | — (ball precision is not measured yet) |

What your manual inputs change: the **camera repair** raises coverage (one 40-s lost stretch went from 14 % to
100 % usable frames on the Canedo segment). **Excluded stretches** remove fake play. **Shot location clicks**
make xG Medium. The **frame check** turns team shape from Low into Medium. The **match sheet** makes goals and
cards High.
