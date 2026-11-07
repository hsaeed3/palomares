# Agent rules

All agents follow CLAUDE.md. Strict rules repeated here:
- **Commit messages:** never include a session URL (e.g. `Claude-Session: https://claude.ai/code/...`) or any other session link in commit messages, co-author trailers, PR bodies or repo files. The only allowed trailer is `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Runtime budget:** before starting any command, estimate how long it will take. Anything expected to run over 10 minutes must be split into resumable batches (each under 10 minutes) or reduced in scope. Do only the work the current deliverable needs, e.g. the 8 litmus frames, not all 219, and widen the scope only after the small run is verified.
