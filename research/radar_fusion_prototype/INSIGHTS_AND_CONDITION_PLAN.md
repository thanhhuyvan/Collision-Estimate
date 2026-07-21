# MVP Insights and Condition-Based Plan

## Evidence from the first real smoke test

The `scene-0061` replay connected real `YOLO → tracker → lead selector → projected radar → fusion`.
It did not produce a collision warning, which is correct for this non-collision scene.

| Observation | Evidence | Risk if ignored |
|---|---|---|
| Detector dropout | No in-corridor lead object in frames 1–3 | A stale or adjacent target could be treated as the lead vehicle. |
| ID switch | `track-0009` car changed to `track-0010` bus at frame 7 | Velocity/range history could be assigned to another object. |
| Range disagreement | Camera proxy and radar range differ materially | Visual box-size range is not safe physical range. |
| Geometric radar match is weak | 8–11 points project into a lead box | Points can still be clutter or another object. |
| Track stability matters | First stable lead reaches age 3 only at frame 6 | A warning from frame 0/1 would be unjustified. |

## What the condition layer does now

The MVP blocks `caution/warning` and returns `uncertain` when any condition below fails:

```text
1. lead is in ego corridor
2. track age >= 3 frames
3. detector confidence >= 0.60
4. radar association confidence >= 0.65
5. lead track ID did not change inside 600 ms
6. radar range did not jump by > 8 m inside 600 ms
7. camera-radar fusion did not report disagreement
8. fused reliability >= 0.60
```

`none` is used for no target / non-closing / not-yet-stable. `uncertain` is used when a target exists but the system cannot safely trust physical TTC.

## Why this is a patch, not the final solution

These rules reduce false confidence; they do not recover a missed object or prove radar identity. They should remain visible config values and be measured per scenario.

## Next upgrade after condition rules

1. Hold a camera track through short detector dropouts without reusing its old radar measurement.
2. Cluster radar points and require the same cluster to support the lead track across 3 frames.
3. Replace the count-based association score with a learned/temporal score only after collecting labels.
4. Evaluate false warning, missed warning, ID switch, and latency on the small scenario set and recorded clips.
