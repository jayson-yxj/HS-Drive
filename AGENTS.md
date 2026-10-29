# HS-Drive repository instructions

Follow [CONTRIBUTING.md](CONTRIBUTING.md). The user replaced the stage-per-branch convention on 2026-10-29.

- `main` is the reproducible research code line; merging code does not promote a model, reference checkpoint, runtime mode or safety claim.
- Use one short-lived branch per independently deliverable feature. Reuse it for fixes, validation and experimental variants; distinguish runs by commit, run_id and configuration.
- Preserve published commit history, failures and conclusions. No force push, history rewrite, blanket branch deletion or private data/weight uploads.
- For the current one-time consolidation, updating main and deleting remote branches require the user's confirmation of the explicit inventory. Do all review and relevant regression work before requesting that confirmation.
- Keep HS/CARLA, PETR and StreamPETR dependencies isolated. Run relevant tests without repeating model training or CARLA experiments merely to organize Git history.
