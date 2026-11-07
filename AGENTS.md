# Agent rules

All agents follow CLAUDE.md. Strict rules repeated here:
- **Commit messages:** never include a session URL (e.g. `Claude-Session: https://claude.ai/code/...`) or any other session link in commit messages, co-author trailers, PR bodies or repo files. The only allowed trailer is `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Runtime budget:** before starting any command, estimate how long it will take. Anything expected to run over 10 minutes must be split into resumable batches (each under 10 minutes) or reduced in scope. Do only the work the current deliverable needs, e.g. the 8 litmus frames, not all 219, and widen the scope only after the small run is verified.
- **Slowdown watch:** the coordinator runs `scripts/ops/watchdog.sh`. It fires when any pipeline process runs longer than 10 min, free RAM drops below 2 GB, or more than 2 heavy steps (21, 21d, 22, 03_dtm, calib register) run at once. When it fires, the coordinator re-plans: it pauses, splits or re-orders the work. Agents check for running heavy steps (`ps`) before starting their own, and use the quick path (`build.py quick`) and `view.py` before any full build.
