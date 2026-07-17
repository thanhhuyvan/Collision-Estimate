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
