# Hackathon MVP — Lead-Object Collision Warning

This is a deliberately small, warning-only camera-radar demo. It answers one question:

> Is the stable lead object in the ego lane closing quickly enough to issue a warning?

## In scope

```text
camera detection/tracking → ego-corridor lead-object gate
radar range/velocity → reliability-aware late fusion
fused TTC → caution/warning/uncertain fallback → UI/event log
```

The MVP does not control braking, perform full-scene fusion, or force a radar match.

See [MASTER_PLAN.md](MASTER_PLAN.md) for milestones, acceptance criteria, risk controls, and pitch plan.
See [INSIGHTS_AND_CONDITION_PLAN.md](INSIGHTS_AND_CONDITION_PLAN.md) for smoke-test findings and the current condition gates.

## Run the offline demo

From repository root:

```powershell
$env:PYTHONPATH = "src"
python hackathon_mvp/run_demo.py
```

It writes `hackathon_mvp/outputs/lead_object_demo.jsonl`. The synthetic replay includes
good radar association, ambiguous association, and a closing lead vehicle.

## Run the small regression dataset

```powershell
$env:PYTHONPATH = "src"
python hackathon_mvp/run_small_dataset.py
```

The four scenarios in `data/lead_object_small/scenarios.json` cover reliable closing,
ambiguous radar, adjacent-lane, and non-closing cases. The runner writes JSONL decisions
and a pass/fail summary CSV.

## Run the camera adapter on a small image sequence

```powershell
$env:PYTHONPATH = "src"
python hackathon_mvp/run_camera_adapter.py --input-dir <CAM_FRONT-images> --weights yolo11n.pt
```

For nuScenes manifests, use `--manifest <manifest.json> --data-root <dataset-root>` to preserve frame order.

The adapter writes a per-frame camera lead-object record. Its `range_proxy_m` is explicitly
approximate; radar must provide physical range and relative velocity before TTC can be trusted.

## nuScenes end-to-end smoke test

```powershell
$env:PYTHONPATH = "src"
python hackathon_mvp/run_nuscenes_smoke.py --data-root <nuScenes-root> --manifest <manifest.json> --weights yolo11n.pt
```

This creates an annotated MP4 and JSONL with real YOLO detections plus projected RADAR_FRONT
returns. Its radar association is geometric-only and intentionally treated as low-confidence MVP logic.

## Replace synthetic inputs during the hackathon

Implement two adapters that create `LeadObjectInput`:

- `camera`: YOLO + IoU tracker output for the in-corridor lead object; include detector and image quality.
- `radar`: a range/closing-speed measurement only after geometry/temporal association passes its gate.

When radar is absent or ambiguous, pass `None` or a low association confidence. The system falls back
to camera and suppresses high-confidence physical TTC warnings.

## Demo checklist

- Render the selected camera box, ego corridor, radar match state, fused range, TTC, reliability, and risk.
- Preserve an event JSONL for every replay.
- Show one positive closing case, one adjacent-lane non-event, and one ambiguous-radar fallback case.
- Benchmark detector latency separately from the sensor/fusion stage.
