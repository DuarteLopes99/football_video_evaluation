#!/bin/bash
# Kept for old shortcuts: the launcher now lives one level up (football_video_evaluation/start.command),
# which also serves the Offside tab (offside_code/).
exec "$(cd "$(dirname "$0")/.." && pwd)/start.command" "$@"
