# Project Layout

## Primary product workspace

- `vision_collision_monitor/` — the official hackathon product: road-facing vision, telemetry simulator, TTC stream, events, reports, and dashboard data.

## Research archive

- `research/radar_fusion_prototype/` — previous radar-camera fusion experiments. This folder is preserved for future research and must not be imported by the vision-only MVP.

## Shared and existing assets

- `src/guardian_perception/` — reusable detector/tracker primitives and earlier research modules. The vision product may reuse detector/tracker contracts only; radar/fusion modules are experimental.
- `data/` — existing local data assets.
- `scripts/` — data preparation and research utilities.
- `outputs/` — earlier top-level replay artifacts.
- `current work/` — preserved working notes; not moved by this reorganization.

## Rule of ownership

New hackathon code, telemetry contracts, tests, dashboard data, and outputs go inside `vision_collision_monitor/`. New radar work goes inside `research/radar_fusion_prototype/` or a future research folder.
