# Stable Baseline — First Replay Result

## Build that was evaluated

```text
YOLO multi-detect
-> Kalman camera tracking
-> one in-corridor lead candidate
-> point-box radar confirmation only for that candidate
-> radar-primary range / closing speed / TTC
-> 3-frame lead stability + 2-frame warning persistence
-> NONE / CAUTION / WARNING / UNCERTAIN with evidence status
```

The camera box-height range proxy is no longer fused against radar range when radar is trusted. Camera establishes object identity/path; radar provides physical TTC. This removes the earlier artificial `sensor disagreement` caused by treating a monocular proxy as a competing range sensor.

## Replay set

All replays use physical nuScenes timestamps (about 500 ms cadence). Video is rendered at 5 FPS only for review.

| Scene | Description | Frames | Detection frames | Lead frames | Radar-confirmed | `UNCERTAIN` | Warnings |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0061 | Turn left / following van | 39 | 38 | 17 | 3 | 0 | 0 |
| scene-0553 | Intersection / pedestrians / truck | 40 | 40 | 19 | 8 | 6 | 0 |
| scene-0796 | Cross intersection / overtaking traffic | 40 | 39 | 32 | 12 | 8 | 0 |
| scene-1077 | Night / bus stop / high speed | 40 | 31 | 23 | 6 | 3 | 0 |
| **Total** | normal-driving stress set | **159** | **148** | **91** | **29** | **17** | **0** |

## What passed

- Replay is deterministic and produces annotated MP4, JSONL, and `summary.csv` for each scene.
- No warning was emitted across the 159 normal-driving frames.
- `UNCERTAIN` is now caused by meaningful gates: radar range jump, radar association below threshold, or a recent lead ID change. It is no longer dominated by camera-vs-radar range disagreement.
- The output distinguishes `no_lead`, `provisional`, `radar_confirmed`, and `uncertain` evidence states.

## What this does not prove

- These scenes are not collision-positive data. Zero warnings is a false-positive check only; it does not measure warning lead time or recall.
- Point-in-box radar confirmation remains a deliberate MVP simplification. It is acceptable only with the current conservative gates and should not be presented as robust multi-object association.
- The run is a laptop correctness replay, not a vehicle-hardware latency benchmark.

## Artifacts

- `outputs/stable_baseline_scene_0061/`
- `outputs/stable_baseline_scene_0553/`
- `outputs/stable_baseline_scene_0796/`
- `outputs/stable_baseline_scene_1077/`

Each folder contains `annotated.mp4`, `decisions.jsonl`, and `summary.csv`.

## Next action before freeze

Add one controlled positive/closing-lead replay. The baseline can be frozen only after it shows the intended sequence: `provisional -> radar_confirmed -> CAUTION -> WARNING`, while retaining the no-warning behavior above.
