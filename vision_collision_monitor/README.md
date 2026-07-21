# Vision-based Collision Risk Monitor

Official hackathon workspace for a vision-only fleet collision-risk monitor.

## Product path

```text
Road-facing video + telemetry simulator
-> object detection and tracking
-> in-path candidate selection
-> vision TTC estimate and confidence
-> TTC stream, risk event clips, trip summary, dashboard records
```

Radar is intentionally not used by this workspace. Previous fusion work is archived at `../research/radar_fusion_prototype/`.

## Folder guide

- `docs/` — requirements, architecture, and acceptance criteria.
- `src/` — vision-only pipeline code.
- `data/telemetry/` — telemetry simulator contract and small replay fixtures.
- `outputs/` — generated logs, reports, clips, and annotated video.
- `tests/` — deterministic unit and replay tests.

## First build target

One road-facing clip and one telemetry CSV must generate:

1. TTC stream in JSONL and CSV.
2. Collision-risk event list with evidence clips.
3. Per-trip summary.
4. Annotated video.
5. Dashboard-ready event and aggregate records.
