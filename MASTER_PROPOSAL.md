# GuardianCo-Pilot: Vision-Only Forward-Collision Warning Baseline

**Status:** Master proposal for the pre-competition baseline  
**Audience:** Team members, technical reviewers, and implementation agents  
**Scope:** Vertical 3 Perception Engine only; recorded forward-facing video; warning-only

## 1. Decision Summary

We will first build a **measurable, vision-only Forward-Collision Warning (FCW) baseline**. It will replay recorded dashcam footage, identify road users in the ego vehicle's forward path, estimate a **relative Time-to-Collision (TTC) proxy**, and generate graded warnings.

The baseline deliberately does **not** control brakes, send CAN commands, estimate certified physical distance, or claim ADAS-grade safety. Its job is to establish a reliable perception-to-risk signal for the future Guardian World Model and Safety Kernel.

The initial detector is lightweight YOLO because it can be developed and tested without NVIDIA hardware. The system will be detector-neutral from the first commit, allowing NVIDIA DashCamNet + DeepStream + NvDCF to be benchmarked later on the same inputs, logic, and metrics.

## 2. Goal and Success Criteria

### Primary goal

From a recorded forward-facing dashcam clip, produce:

1. an annotated MP4 showing detections, track IDs, ego corridor, selected collision candidate, TTC proxy, confidence, warning state, and latency; and
2. structured JSONL/CSV output containing every frame's inference and warning decision.

### Baseline success

The baseline is considered complete when it can replay a fixed scenario suite and:

- identify the relevant lead target in the ego corridor for the majority of manually labelled scenarios;
- avoid warning for clear adjacent-lane and parked-object negative cases;
- never issue a warning from a single-frame detection or unstable target;
- produce reproducible event logs when replaying the same clip and configuration;
- measure p50, p95, and p99 capture-to-risk latency, with **p95 < 100 ms** as the target on the future GPU PC; and
- clearly record false warnings, missed close approaches, and their known causes.

The target is an evidence-backed prototype, not a guarantee that it prevents every collision.

## 3. Fixed Scope and Non-Goals

### In scope

- One forward RGB camera stream, processed from recorded files.
- Vehicles, motorcycles, bicycles, and persons as relevant classes.
- Detection, multi-object tracking, ego-path relevance, temporal smoothing, visual TTC, confidence, and graded warning output.
- Offline data replay, video annotation, structured metrics, and failure review.
- Future detector/runtime comparison: YOLO versus NVIDIA DashCamNet/DeepStream/NvDCF.

### Explicitly out of scope for v1

- Live camera capture, CAN telemetry, radar, LiDAR, stereo depth, GNSS, and vehicle control.
- Driver monitoring, lane-departure warning, road-surface estimation, dashboard/HMI, LLM explanations, or Safety-Kernel integration.
- Model training or fine-tuning from scratch.
- Automatic braking, steering, or any safety-critical actuation.

**Reason:** These are useful future capabilities, but each creates new data, hardware, validation, and safety work. Including them now would prevent a stable collision-warning baseline from being finished and measured.

## 4. Baseline Pipeline

```text
Recorded video
  -> Decode and timestamp
  -> Detector adapter (YOLO now; NVIDIA later)
  -> Multi-object tracker
  -> Ego-corridor relevance filter
  -> Temporal state and smoothing
  -> Visual TTC proxy + confidence
  -> Risk decision and hysteresis
  -> Annotated video + JSONL/CSV metrics
```

### 4.1 Detector adapter

All detector implementations must return the same record:

```text
Detection(frame_id, timestamp_ms, bbox_xyxy, class_name, detector_confidence)
```

YOLO is the first adapter. The later NVIDIA adapter maps DeepStream metadata from DashCamNet to exactly this schema.

**Reason:** Collision logic should not depend on a particular model or GPU runtime. This makes the NVIDIA decision a benchmark, not a rewrite.

### 4.2 Tracking

Tracking assigns a persistent `track_id` to detections. A new target remains **tentative** until it has appeared consistently for a configured probation window (initially 10 frames). A target with uncertain identity, poor confidence, or a long detector gap cannot trigger a warning.

**Reason:** A detector box has no motion history. TTC and collision risk require the same object to be observed over time. This also prevents one-frame false positives and reduces alert fatigue.

### 4.3 Ego-corridor relevance

Use an editable normalized polygon describing the expected forward driving corridor. A track is a collision candidate only when its bottom-centre point remains inside this polygon.

**Reason:** The visually largest or nearest object can be in an adjacent lane. A fixed corridor is a low-cost first approximation; later versions can replace it with lane segmentation or inverse-perspective geometry.

### 4.4 Visual TTC proxy

For each stable candidate, maintain a rolling 10-frame history of smoothed bounding-box height. For a roughly constant-size object approaching along the optical axis, increasing image height implies decreasing range. Estimate a visual TTC proxy from the positive growth rate. This is not a metre-accurate range estimate.

Initial output fields:

```text
track_id, ttc_estimate_s, ttc_confidence, approach_rate, risk
```

Use `none`, `caution`, and `warning` risk states. Initial thresholds are `caution <= 4.0 s` and `warning <= 2.5 s`, subject to scenario-suite tuning. Warnings require stable low TTC across multiple frames; recovery requires a separate higher exit threshold (hysteresis).

**Reason:** Raw frame-by-frame box size is noisy. Smoothing, confidence gates, persistence, and hysteresis are more important for a useful baseline than choosing a more complex detector.

### 4.5 Output and observability

The annotated video shows all decisions. The JSONL/CSV log stores per-stage latency, selected target, confidences, TTC, warning state, and configuration version. A summary report calculates p50/p95/p99 latency, warning count, false warnings per minute, misses, ID switches, and warning lead time.

**Reason:** A model demo is not sufficient evidence. The logs make every warning reviewable, reproducible, and comparable across models.

## 5. Data and Virtual-Environment Strategy

Use real and simulated data for different purposes. Neither source is sufficient alone.

| Source | Use | Reason | Limitation |
|---|---|---|---|
| BDD100K | Detection/tracking robustness | Diverse driving video plus detection, lane, and tracking tasks | No direct collision-warning ground truth |
| Nexar Collision Prediction | Main real-world warning benchmark | Collision/near-collision and normal clips with event/alert-time labels | Dataset domain may differ from Vietnamese traffic |
| DADA-2000 | Accident and visibility stress tests | Day/night/weather variety and annotated accident windows | Not all events are forward lead-target situations |
| KITTI | Geometry experiments | Calibrated camera sequences and tracked objects | Older, geographically narrow domain |
| CARLA 0.9.15 | Controlled scenario validation | RGB camera, exact actor states, collision events, weather, traffic, scripted scenarios | Synthetic-to-real domain gap |

Start with a curated suite, not entire datasets:

- 20-30 BDD100K clips for detection/tracking cases;
- 50-100 Nexar clips as the core warning benchmark;
- 10-20 DADA-2000 clips for accident/visibility stress;
- a small set of KITTI sequences only when geometry work starts.

Use CARLA 0.9.15 after the replay baseline exists. Record RGB video together with hidden evaluator data: collision sensor events, actor transforms, and depth. Run the baseline from RGB replay only; use hidden data exclusively to evaluate it.

**Reason:** Real clips show actual camera-domain failures. CARLA creates controlled cut-ins, hard braking, adjacent-lane negatives, rain, curves, and collision timestamps that are rare and hard to label in real video. Using simulator-only depth or pose in inference would violate the vision-only goal.

## 6. Initial CARLA Scenario Suite

Record deterministic scenarios with fixed seeds and three visibility settings where supported:

1. Lead vehicle brakes hard in the ego corridor.
2. Stopped vehicle is revealed ahead.
3. Vehicle cuts into the ego corridor.
4. Adjacent-lane vehicle closes but must not generate a warning.
5. Pedestrian or motorbike crosses the ego path.
6. Curved-road or partial-occlusion case.

For each scenario, save the RGB video, simulator seed, target actor ID, collision/no-collision outcome, event timestamp, and evaluator-only trajectory/relative state.

**Important:** Record simulator video first, then replay it for model benchmarking. Do not run CARLA and inference concurrently when measuring FCW latency because they compete for GPU resources.

## 7. Risk Register and Controls

| Risk | Impact | Control in baseline |
|---|---|---|
| Adjacent/parked object triggers warning | Loss of driver trust | Ego corridor, target persistence, manual negative scenarios |
| Motorbike, pedestrian, night, rain, glare, occlusion causes miss | Late/no warning | Diverse regression clips, class-specific review, confidence reporting |
| Curves, hills, pitch changes distort TTC | Incorrect risk estimate | Visual TTC only, instability suppression, later geometry/horizon upgrade |
| Tracker ID switch corrupts history | Wrong TTC/warning | Tentative/confirmed state, reset history on low confidence or identity uncertainty |
| Cut-in appears too late | Insufficient warning lead time | Separate cut-in scenarios and a new-in-path state; report lead time |
| GPU/CPU/display/write causes latency spikes | Unsafe or misleading timing result | Per-stage p50/p95/p99 measurement; keep rendering/export outside risk timing |
| CARLA realism differs from Vietnam | Overfit to simulation | Use simulation for logic tests only; use real dashcam clips for primary evaluation |
| YOLO-to-NVIDIA migration becomes rewrite | Schedule risk | Detector-neutral contract and shared scenario/metric suite |
| Scope expands before evidence exists | Baseline never completes | Enforce v1 non-goals and milestone gates |
| Model/package license conflict | Submission risk | Record model/package version and verify competition/publication terms before release |

## 8. Hardware and Runtime Assumptions

The implementation laptop is used for planning, documentation, source control, and lightweight work. The baseline runs on a separate Intel + NVIDIA GPU PC.

Recommended planning target:

- Intel Core i7-class CPU, 6+ cores;
- NVIDIA RTX GPU with 12 GB VRAM preferred; 8 GB is the practical minimum for lightweight replay inference;
- 32 GB RAM;
- 1 TB NVMe SSD; and
- Ubuntu 22.04 for the cleanest future NVIDIA TensorRT/DeepStream path.

CARLA 0.9.15 should use a packaged release, not a source build, during baseline work.

**Reason:** This preserves local laptop resources, lets the pipeline target a realistic edge/desktop GPU, and avoids spending early time on simulator compilation or NVIDIA deployment before the core logic is proven.

## 9. Milestones, Deliverables, and Time Estimate

| Milestone | Deliverable | Exit criteria | Estimate |
|---|---|---|---:|
| M0: Data and evaluation setup | Scenario manifest and small fixed clip suite | Every clip has expected target and warning/no-warning label | 0.5-1 day |
| M1: Detection replay | Annotated detection video | Stable file input and model outputs with stage timing | 1 day |
| M2: Tracking/relevance | Track IDs and ego-corridor target selector | Adjacent-lane negative case is rejected | 1-1.5 days |
| M3: TTC/risk | Smoothed TTC, confidence, and warning states | No one-frame warning; event logs generated | 1-1.5 days |
| M4: Evidence baseline | Metrics report and reviewed regression suite | Reproducible results and p95 latency report | 1-2 days |
| M5: CARLA validation | Recorded controlled scenarios and evaluator report | Risk logic tested against known scenario outcomes | 1-2 days |
| M6: NVIDIA A/B | DashCamNet/DeepStream/NvDCF benchmark | Keep NVIDIA only if it improves measured metrics | 1-3 days after hardware/software access |

**Expected usable baseline:** 5-7 focused working days for M0-M4.  
**Expected credible baseline:** 7-10 focused working days including CARLA validation and failure review.  
**NVIDIA comparison:** separate, optional after the baseline is stable.

## 10. Evaluation Protocol

For every configuration and clip, record:

- detector and tracker version;
- video ID, camera configuration, and scenario label;
- p50/p95/p99 capture-to-risk latency and latency per stage;
- relevant-target detection recall;
- track-ID switches;
- false warnings per minute;
- missed close approaches;
- warning lead time before labelled event; and
- manual reviewer note for each warning/miss.

Never accept an improvement based only on an attractive overlay. Retain a change only when it improves the selected metric without materially worsening the fixed regression suite.

## 11. Decision Gates and Future Upgrades

### Gate A: Baseline readiness

Begin M1 when the GPU PC has NVIDIA drivers and Python working, the curated replay suite is available, and a scenario manifest exists. CARLA is not a blocker.

### Gate B: Geometry upgrade

Add camera calibration, virtual horizon, lane segmentation, or inverse-perspective mapping only if false warnings/misses are primarily caused by curves, pitch changes, or poor ego-path relevance.

### Gate C: NVIDIA adoption

Use DashCamNet/DeepStream/NvDCF only if the same fixed suite shows an improvement in relevant-target recall, ID stability, false-warning rate, or p95 latency sufficient to justify the deployment complexity.

### Gate D: Cross-vertical integration

Only after a stable M4 result, publish the detector-neutral Scene Graph-lite event to Vertical 4. The future payload should contain target class, track ID, normalized geometry, TTC proxy, confidence, risk state, and timing metadata. Vertical 4 retains final safety authority.

## 12. References

- BDD100K dataset: https://github.com/bdd100k/bdd100k
- Nexar Collision Prediction dataset: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
- Nexar dataset paper: https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge.html
- DADA-2000: https://arxiv.org/abs/1904.12634
- KITTI Vision Benchmark: https://www.cvlibs.net/datasets/kitti/
- CARLA sensors: https://carla.readthedocs.io/en/latest/ref_sensors/
- CARLA ScenarioRunner: https://scenario-runner.readthedocs.io/en/latest/
- CARLA 0.9.15 release: https://github.com/carla-simulator/carla/releases
- NVIDIA DeepStream tracker robustness: https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Accuracy.html
