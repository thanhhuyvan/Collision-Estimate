# Current Work: Latency-First Collision Warning Baseline

## Current state

The vision-only baseline is functionally complete for offline investigation:

```text
video → YOLO detector → IoU tracker → ego corridor → visual TTC/risk → warning evidence
```

It runs on the six local Nexar clips and writes annotated MP4s, per-frame JSONL decisions,
and a CSV summary. It is not yet a vehicle-ready collision-warning system.

## Measured laptop limitation

All measurements below use one 1280x720 replay frame, YOLO image size 640, CPU PyTorch,
five warm-up iterations, and 30 measured iterations. Values are warm p95 inference latency.

| Backbone | Model size | Warm p95 | Estimated full Fast Path | Status |
|---|---:|---:|---:|---|
| YOLO11n | 5.4 MiB | 88.2 ms | 148.2 ms | Laptop functional test only |
| YOLO11s | 18.4 MiB | 348.6 ms | 408.6 ms | Too slow on laptop CPU |
| YOLO11m | 38.8 MiB | 603.2 ms | 663.2 ms | Too slow on laptop CPU |

The proposal requires less than 100 ms end to end, with about 35 ms reserved for perception
inference. The current Python + Ultralytics + CPU PyTorch path cannot meet this requirement.
This does not prove that YOLO11n is unsuitable; it proves that this laptop execution stack is
not the vehicle deployment stack.

## Detection-quality issues found

The initial six-clip investigation produced no false warning on the three negative clips, but
missed all three positive clips:

| Clip | Preliminary failure category | Evidence |
|---|---|---|
| 00015 | Threshold issue | Detections and a persistent in-corridor track existed; minimum visual TTC was 2.864 s, above the current 2.5 s warning threshold. |
| 00026 | Detector miss | No detections in the evaluation window. |
| 00054 | Corridor error | Detections existed, but none entered the fixed ego corridor. |

Other known limitations:

- The tracker is greedy same-class IoU only. It has no learned appearance model or motion
  prediction, so occlusion and dense traffic can fragment IDs.
- The ego corridor is a fixed image-space trapezoid. It needs camera/lane calibration before
  it can reliably handle curves, cut-ins, and different camera mounting positions.
- TTC is a 2D bounding-box-growth proxy. It is not physical distance or relative speed.
- Six clips are enough for plumbing and failure analysis, not a safety or accuracy claim.

## Latency target for the GPU baseline

Use p95 and p99 capture-to-warning latency, not only average FPS. The practical baseline gate
is p95 at or below 80 ms and p99 below 100 ms, leaving margin below the proposal requirement.

| Fast Path stage | Target budget |
|---|---:|
| Camera capture / hardware decode / transfer | 15 ms |
| Detector plus post-processing | 30 ms p95 |
| Tracking and World Model update | 5 ms |
| Corridor, TTC, risk, and Safety Kernel | 3 ms |
| Warning event / CAN or HMI hand-off | 10 ms |
| Scheduling and tail-latency margin | 17 ms |

## Next steps

### 1. Establish the NVIDIA GPU baseline

On the target NVIDIA PC or Jetson Orin NX:

1. Install CUDA-compatible PyTorch, TensorRT, and the matching GPU driver / JetPack stack.
2. Run `scripts/benchmark_backbones.py --device 0 --weights yolo11n.pt`.
3. Record warm p50, p95, p99, GPU clock, temperature, power mode, and memory use.
4. Reject the candidate from Fast Path if detector p95 exceeds 30 to 35 ms.

### 2. Optimize the accepted detector

1. Export YOLO11n to ONNX.
2. Build a fixed-shape TensorRT FP16 engine and benchmark it.
3. Build an INT8 engine with representative driving images for calibration.
4. Compare detections on the six clips before accepting INT8.
5. Try YOLO11s only if YOLO11n still misses relevant targets and YOLO11s meets the same
   latency gate.

### 3. Integrate a true Fast Path

Replace the offline OpenCV replay path with:

```text
camera → GStreamer/DeepStream hardware decode → TensorRT detector
       → tracker → TTC/risk → warning event
```

Use batch size one, fixed input dimensions, bounded queues, and stale-frame dropping. Pre-warm
the engine before driving. Never allow a waiting queue to turn high throughput into old-frame
warnings.

### 4. Fix the evidence-backed quality failures

1. Review `00015` and evaluate conservative warning-threshold alternatives.
2. Inspect `00026` at several input resolutions and detector confidence thresholds.
3. Replace or calibrate the fixed corridor for `00054` before changing TTC logic.
4. Upgrade to NvDCF or a motion-aware tracker only if logs show that ID fragmentation is a
   material source of misses.

### 5. Prove the end-to-end result

For every GPU build, preserve:

- per-component and capture-to-warning p50/p95/p99 latency;
- annotated replay and JSONL evidence;
- warning lead time relative to Nexar event time;
- false warnings on negative clips; and
- thermal/power condition during the benchmark.

The first decision point is simple: determine whether YOLO11n TensorRT meets the detector
latency gate and detects the relevant target. Do not introduce a heavier backbone until this
answer is measured.

## Current pivot: reliability-aware sensor fusion

The first fusion baseline is intentionally late/object-level fusion rather than a learned BEV
network. `guardian_perception.fusion` fuses a camera track's estimated range/closing speed with
an already-associated radar return. Camera detector confidence and image quality determine the
camera reliability; radar signal quality and association confidence determine radar reliability.
Radar is rejected below the association gate, and meaningful sensor disagreement halves fused
reliability instead of silently producing an over-confident TTC.

Run the dependency-free learning demo with:

```powershell
$env:PYTHONPATH = 'src'
python scripts\run_reliability_fusion_demo.py
```

The next dataset step is to replace the demonstration measurements with timestamped nuScenes
camera/radar/ego data, while retaining every fusion weight and fallback explanation in the log.

### Local nuScenes fusion subset

The local `E:\datasets\nuscenes\mini` dataset subset contains eight synchronized CAM_FRONT
images and RADAR_FRONT point clouds from `scene-0061`. Generate its compact manifest with:

```powershell
python scripts\prepare_nuscenes_fusion_subset.py `
  --data-root E:\datasets\nuscenes\mini `
  --scene scene-0061 --frames 8
```

The manifest contains the source timestamps, camera/radar calibration, ego poses, and ground
truth annotations needed for association and late-fusion experiments. It does not use ground
truth as a runtime fusion input.

Render an inspection video before implementing association:

```powershell
& .\.venv\Scripts\python.exe scripts\render_nuscenes_fusion_preview.py `
  --data-root E:\datasets\nuscenes\mini
```

The resulting video overlays projected radar points and evaluation-only ground-truth 3D boxes
on CAM_FRONT, then places raw RADAR_FRONT points in Bird's-Eye View. It is a diagnostic tool
for calibration, timestamp offset, radar clutter, and association ambiguity.
