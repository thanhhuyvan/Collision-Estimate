# Insight Log — Guardian Co-Pilot

Mục đích: lưu các kết luận quan trọng có bằng chứng để các thử nghiệm sau không
lặp lại giả định cũ. Mỗi entry phải phân biệt rõ **evidence**, **phạm vi**, và
**quyết định**; không biến kết quả trên subset nhỏ thành khẳng định deploy được.

## Insight tổng hợp từ quá trình trao đổi

### 2026-07-21 — Mục tiêu trước mắt là usable baseline, không phải SOTA detector

**Evidence**

- Mục tiêu được thống nhất nhiều lần là hiểu pipeline, failure modes và khả năng
  triển khai trước khi competition bắt đầu.
- Training, fine-tuning, braking control và safety certification đều nằm ngoài
  scope laptop test hiện tại.

**Insight**

Baseline tốt phải là baseline có contract rõ, chạy lại được, có evidence video/
JSONL, và chỉ ra failure; không phải baseline có con số cao nhưng không biết nó
đến từ detector, association hay threshold.

**Decision**

Mọi thay đổi sau này phải có baseline comparison, held-out evidence và failure
classification. Không đổi nhiều thành phần cùng lúc.

### 2026-07-21 — Latency là gate triển khai, không phải KPI duy nhất của research

**Evidence**

- Laptop CPU benchmark cho YOLO11n warm p95 khoảng 88 ms inference và fast path
  ước tính khoảng 148 ms; YOLO11s/m chậm hơn nhiều.
- Proposal vehicle target đòi hỏi end-to-end dưới 100 ms, cần GPU deployment
  stack như TensorRT/DeepStream thay vì Python + CPU replay.

**Insight**

Laptop phù hợp để kiểm tra correctness, data contract và ablation; nó không đại
diện vehicle hardware. Một backbone nặng hơn không tự làm project thất bại nếu
target PC/Jetson có GPU phù hợp, nhưng chỉ được chọn sau khi đo p95/p99 trên target.

**Decision**

Tách hai benchmark: research-quality/offline và deployment-latency. Không suy
diễn latency vehicle từ laptop CPU, cũng không dùng accuracy offline để bỏ qua
latency budget.

### 2026-07-21 — Không có free lunch: phải đóng Operational Design Domain (ODD)

**Evidence**

- Mục tiêu vừa muốn generalize mọi scene, accuracy cao, latency thấp và hardware
  nhẹ đã tạo conflict trong các thử nghiệm dense traffic.
- Radar sparse/multipath, camera occlusion và scene distribution shift đều tồn tại.

**Insight**

Không có một model nhỏ, tổng quát cho mọi traffic/weather/hardware mà vẫn bảo đảm
collision performance. Độ tin cậy đến từ scope/ODD rõ ràng, fallback và uncertainty
management chứ không chỉ từ backbone.

**Decision**

Baseline collision scope được định nghĩa là CAM_FRONT + RADAR_FRONT, lead/cut-in
trong ego corridor, một range/ODD được công bố rõ, và output warning-only. Trường
hợp ngoài confidence phải abstain/camera-only, không tạo TTC giả.

### 2026-07-21 — Camera và radar không phải hai bảng dữ liệu ghép trực tiếp

**Evidence**

- Camera trả semantic/appearance/2D box; radar trả sparse range, relative velocity
  và clutter nhưng không có semantic class trực tiếp.
- Alignment phụ thuộc timestamps, ego pose, extrinsic/intrinsic calibration và
  uncertainty; 2D overlap đơn thuần tạo nhiều false association.

**Insight**

Fusion không phải nối cột camera với radar. Nó là bài toán đưa measurement về cùng
coordinate/time frame, sinh candidate, ước lượng correspondence confidence, rồi mới
fusion measurement.

**Decision**

Giữ pipeline theo thứ tự `synchronise → calibrate/project → candidate/associate →
confidence/abstain → fuse → risk`; không cho radar đi thẳng vào TTC từ raw point.

### 2026-07-21 — Ground truth là công cụ học và đánh giá, không phải runtime input

**Evidence**

- nuScenes GT 3D boxes/instance IDs giúp tạo label và audit cluster overlap.
- Các demo oracle box được dùng để cô lập association, nhưng không phản ánh YOLO
  hoặc detector thật.

**Insight**

Oracle experiment hữu ích để localize lỗi, nhưng kết quả oracle không phải kết quả
deploy. Leakage giữa GT và runtime sẽ tạo cảm giác model tốt giả tạo.

**Decision**

Mọi artifact phải ghi rõ `oracle diagnostic` hay `runtime-compatible`. GT bị khóa
ở label/evaluation layer; pipeline runtime thay bằng detector/tracker thật.

### 2026-07-21 — Dataset metadata đủ để bắt đầu, nhưng label radar là derived label

**Evidence**

- nuScenes có synchronized sensor data, calibration, ego pose, 3D annotations và
  instance tracks; không cung cấp semantic instance label hoàn hảo cho từng radar
  return.
- Cluster labels được suy ra bằng overlap với GT 3D box.

**Insight**

Radar-object label cần uncertainty: positive, negative và ignore. Ép mọi return
thành nhãn nhị phân làm bẩn supervision, đặc biệt trong multipath/biên object.

**Decision**

Preserve `ignore` cho audit và loại khỏi fitting; mở rộng sequence/diversity trước
khi tin bất kỳ learned model result nào.

### 2026-07-21 — Safety behaviour ưu tiên biết từ chối hơn coverage cao

**Evidence**

- Single-sweep baseline có coverage 80–100% ở một số scene nhưng precision có thể
  xuống 0–22%.
- Sai radar target dẫn đến sai range/range-rate/TTC, nghiêm trọng hơn missing một
  radar confirmation trong camera-led warning.

**Insight**

Với forward collision warning, false confident fusion nguy hiểm hơn radar missing
return. Coverage không phải objective độc lập.

**Decision**

Chỉ fuse radar khi association confidence được chứng minh. Các trạng thái hợp lệ là
`fused`, `camera-only`, và `unknown/abstain`; mọi report phải tách coverage, precision
và false-warning impact.

## 2026-07-21 — Phải tách collision warning khỏi generic radar–camera association

**Evidence**

- Full multi-object association trên held-out dense scene cho precision thấp, dù
  calibration và GT object labels đã khả dụng.
- Collision scope chỉ cần lead/cut-in target trong ego corridor, không cần gán
  semantic identity cho toàn bộ radar point/cluster trong scene.

**Insight**

`full-scene association` và `forward collision warning` là hai bài toán khác
nhau, có metric và failure mode khác nhau. Dùng metric full-scene làm KPI chính
cho collision warning sẽ làm sai hướng tối ưu.

**Decision**

Đánh giá collision pipeline bằng lead-target correctness, range error, TTC
stability, false/missed warning và abstention; giữ full-scene association như
diagnostic riêng.

## 2026-07-21 — Single-frame 2D projection score không an toàn trong scene khó

**Evidence**

- Collision-target test 40-frame thay đổi mạnh giữa scene: radar precision khi
  được dùng từ 0% đến 91.9%.
- Tuning threshold không tạo được điểm vận hành an toàn: giảm threshold chỉ tăng
  coverage sai; tăng threshold làm mất recall mà precision vẫn thấp ở hard scene.

**Insight**

Projection overlap/box-center score không thể hiện uncertainty của radar clutter,
occlusion hoặc object overlap. Radar association confidence hiện bị over-confident.

**Decision**

Không fuse radar vào TTC chỉ vì một cluster có support trong 2D box. Runtime phải
cho phép `abstain`/camera-only và log lý do từ chối.

## 2026-07-21 — Multi-sweep không tự động tạo temporal association

**Evidence**

- Controlled 32-frame ablation trên 7 lead frames: single-sweep và 3-sweep +
  Doppler đều đạt 71.4% precision/safe success.
- 3-sweep tăng số candidate/positive label nhưng không đổi lead-target result.

**Insight**

Thêm radar points qua thời gian chỉ tăng density. Nó không giải quyết ambiguity
nếu scorer không có representation/tracklet đủ mạnh để dùng motion và Doppler
nhằm phân biệt các object.

**Decision**

Không tiếp tục extract hoặc tune multi-sweep chỉ để kỳ vọng cải thiện. Temporal
fusion chỉ được mở lại khi có một association contract/tracklet model rõ ràng.

## 2026-07-21 — Lightweight temporal feature scorer chưa có cải thiện ổn định

**Evidence**

- Experiment 1,649 labelled candidates, leave-scene-out:
  - scene-0061: precision 55.6% → 61.9%, nhưng safe success 51.7% → 44.8%;
  - scene-0553: giữ 100% trên chỉ 9 lead frames;
  - scene-0796: giữ 0% trên 5 lead frames;
  - scene-1077: giữ 94.6% / 89.7%.
- Extra features là camera box-growth, camera-range-proxy residual, radar Doppler
  closing và closing-growth agreement.

**Insight**

Các proxy hiện tại không mang signal tổng quát đủ mạnh. Một gain cục bộ đi kèm
coverage/safe-success giảm không được xem là improvement của baseline.

**Decision**

Dừng feature stacking và không coi logistic scorer là hướng baseline hiện tại.
Trước thử nghiệm mới, định nghĩa lại một association contract duy nhất hoặc quay
về collision-warning pipeline camera-led với radar chỉ là measurement confirmation.

## 2026-07-21 — Data contract và calibration không phải blocker hiện tại

**Evidence**

- nuScenes cung cấp camera/radar timestamps, ego poses, sensor calibration, GT
  3D object boxes và instance IDs.
- Đã tạo 159-frame manifest, candidate labels `positive/negative/ignore`, cùng
  audit video. Unit tests 19/19 pass.

**Insight**

Blocker hiện tại là ambiguity/representation của association và quy mô diversity
của sequence, không phải thiếu raw metadata hoặc lỗi calibration mặc định.

**Decision**

Giữ label/audit/evaluator làm infrastructure tái sử dụng; không dùng GT trong
runtime, chỉ dùng để sinh label và đánh giá.

## 2026-07-21 — Quyết định thử nghiệm tối ưu hoá: physics-constrained learned tracklet association

**Evidence**

- Hard single-frame rule association không generalize ổn định giữa các scene.
- Thêm multi-sweep hoặc proxy feature riêng lẻ không tạo improvement nhất quán.
- Collision warning cần uncertainty-calibrated range/range-rate cho một số ít
  risk-relevant targets, không cần generic full-scene 3D detection là mục tiêu chính.

**Insight**

Bài toán tối ưu hoá nên là học likelihood association ở cấp **camera tracklet ↔
radar tracklet**, sau khi physics/calibration gate đã giới hạn candidate. Learned
model xử lý semantic/appearance residual; Bayesian tracker xử lý dynamics,
multi-hypothesis và uncertainty.

**Decision**

Canonical experiment là **Physics-Constrained Learned Association + UKF/JPDA**:

1. UKF dự đoán lead/cut-in object state và covariance;
2. Mahalanobis/frustum gate tạo candidate radar tracklets;
3. learned scorer (FiLM là một ablation) trả association likelihood và measurement
   uncertainty;
4. JPDA cập nhật state bằng weighted hypotheses;
5. TTC posterior quyết định `warning`, `camera-only`, hoặc `abstain`.

So sánh bắt buộc: camera-only, hard assignment + UKF, PDAF/JPDA thuần vật lý,
learned association không physics, và hybrid. Không chấp nhận improvement chỉ ở
một scene; metric chính gồm association calibration, lead-target correctness,
range/range-rate error, TTC calibration và false/missed warning.

## 2026-07-21 — Classical probability replaces hard decisions; it does not remove all rules

**Evidence**

- Hard 2D score/threshold created high coverage but potentially 0% precision in
  difficult scenes.
- Adding isolated temporal heuristics did not generalize.

**Insight**

UKF/JPDA/PDAF move the decision from `one score → one forced match` to a posterior
over state and association hypotheses. The expected first gain is better-calibrated
uncertainty and more valid abstention, not an automatic increase in raw coverage.

**Decision**

Retain deterministic rules only as non-negotiable safety/data-validity constraints:
timestamp bounds, calibration validity, physical range/velocity limits, ODD and
warning policy. Use probabilistic state/association for measurement fusion; use
learned FiLM/MLP only as a bounded likelihood correction. Treat reduced coverage
as a success only when false fusion and TTC/warning error decrease on held-out data.
