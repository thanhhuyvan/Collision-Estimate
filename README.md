# Guardian Perception Core

CPU-only foundation for the GuardianCo-Pilot vision baseline. It intentionally contains no
detector, tracker, OpenCV, CUDA, or DeepStream dependency.

## What runs on the laptop

- stable data contracts for detector and tracker adapters;
- ego-corridor target relevance;
- temporal TTC proxy and confidence gating;
- warning hysteresis; and
- structured JSONL event logging.

The future YOLO and NVIDIA/DeepStream adapters supply `TrackObservation` objects to the
same `RiskEngine`. They do not own the risk logic.

## Run the tests

```powershell
$py = 'C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m unittest discover -s tests -v
```

## Optional laptop YOLO smoke test

The laptop environment is intentionally CPU-only. Create its isolated environment and install
the optional detector dependency with:

```powershell
$py = 'C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install ultralytics
```

`guardian_perception.adapters.YoloDetector` converts YOLO output into `Detection` records.
It does not track objects or make risk decisions. The future tracker adapter must emit
`TrackObservation` records, which are the only input accepted by `RiskEngine`.

CPU inference is useful for functional smoke tests only. Run real-time video, GPU latency, and
NVIDIA DeepStream/TensorRT benchmarks on the target GPU PC.

## Six-clip end-to-end laptop test

`data/nexar_samples/metadata.csv` is the deliberately small local manifest for the six
downloaded Nexar clips. The event, alert, weather, lighting, and scene fields are reference
metadata only: they are never supplied to YOLO, the tracker, or the risk engine. Keep raw
Nexar videos local under `data/nexar_samples/positive` and `data/nexar_samples/negative`;
they are excluded from Git because of their source licence.

With `yolo11n.pt` in the repository root, run the complete offline investigation at 5 FPS:

```powershell
$env:PYTHONPATH = 'src'
& .\.venv\Scripts\python.exe scripts\check_laptop_setup.py
& .\.venv\Scripts\python.exe scripts\run_laptop_baseline.py
```

The runner uses positive windows from `alert - 5 s` through `event + 2 s`, and negative
windows from 10 to 20 seconds (or to end-of-file when a supplied negative clip is shorter).
It writes the following under `outputs/laptop_test/`, which is intentionally ignored by Git:

- one `annotated.mp4` and per-frame `decisions.jsonl` for each clip; and
- one `summary.csv` with warning timing, TTC, detector/tracker counts, reference metadata,
  and a preliminary diagnostic category for every missed positive.

The tracker is a dependency-free same-class greedy-IoU tracker (`>= 0.30`) with a 400 ms
missed-observation expiry. It emits the existing `TrackObservation` contract, so it can later
be swapped for ByteTrack or NVIDIA NvDCF without changing ego-corridor or risk logic.

## Backbone latency gate

Never choose a heavier detector from accuracy alone. The proposal reserves 35 ms for
perception inference and 100 ms for the complete Fast Path. Measure candidate backbones on
the target device with a warm p95 latency gate:

```powershell
$env:PYTHONPATH = 'src'
# Laptop comparison: functional estimate only
& .\.venv\Scripts\python.exe scripts\benchmark_backbones.py --weights yolo11n.pt yolo11s.pt yolo11m.pt
# NVIDIA PC: the decision-quality benchmark
& .\.venv\Scripts\python.exe scripts\benchmark_backbones.py --device 0 --weights yolo11n.pt yolo11s.pt yolo11m.pt
```

The script writes JSON and CSV evidence under `outputs/backbone_benchmarks/`. A backbone is
accepted for Fast Path only when its measured warm p95 is at most 35 ms; its estimated total
is the p95 inference plus the proposal's fixed 60 ms allowance for capture, planning, safety,
and UI/CAN. The laptop result must not be used as the final vehicle decision.
