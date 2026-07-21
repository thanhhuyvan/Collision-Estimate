# Hackathon MVP Stability Plan

## Goal

Deliver a camera-radar collision-warning demo that is explainable, conservative, and repeatable on a small test set. This is an engineering MVP, not a multi-object fusion research project.

## Frozen MVP pipeline

```text
YOLO multi-detect
  -> lightweight Kalman camera tracking
  -> select one lead vehicle in the ego corridor
  -> radar confirmation only when association is clear
  -> TTC/risk decision with hysteresis
  -> WARNING / CAUTION / UNCERTAIN / NONE
```

YOLO continues to detect every visible object. Only one in-path lead object enters the collision decision. Ambiguous radar is never forced onto a camera target.

## Safety policy

- Require a stable camera lead for at least 3 frames.
- Use radar range and relative velocity only when its association passes the configured confidence gate.
- If radar is absent or ambiguous, return `CAMERA_ONLY`/`UNCERTAIN`; do not invent a radar measurement.
- Emit `WARNING` only with stable lead, reliable radar, and low TTC.
- In dense traffic, prefer `UNCERTAIN` over a false warning.

## Three acceptance scenarios

| Scenario | Expected behavior | Purpose |
|---|---|---|
| Single lead on a straight path | Stable lead and sensible risk progression | Main demo |
| Brief camera dropout | Keep/recover identity without a warning spike | Temporal robustness |
| Dense or ambiguous traffic | No forced association; `UNCERTAIN` or `NONE` | Safety behavior |

## Short execution plan

1. Select one short nuScenes clip for each acceptance scenario and save annotated video plus JSONL evidence.
2. Tune only camera stability, radar association, TTC, and hysteresis thresholds against those clips.
3. Freeze the default pipeline after it passes all three scenarios; do not add a heavier backbone.
4. Build the demo screen/video: all detections, selected lead, radar status, TTC, and risk state.
5. Keep global-frame temporal association and multi-object fusion as post-hackathon experiments, not MVP dependencies.

## Definition of done

- The same clips produce the same result on repeated runs.
- No warning is emitted before the 3-frame stability gate.
- Ambiguous scenes show `UNCERTAIN` rather than an unjustified warning.
- A reviewer can see why every warning or non-warning occurred from the annotated video and JSONL log.
