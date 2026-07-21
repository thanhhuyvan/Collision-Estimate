# Telemetry Simulator Contract

One row represents the ego vehicle state at a video-aligned timestamp.

```csv
timestamp_ms,vehicle_id,trip_id,ego_speed_mps,brake_active,steering_angle_deg,latitude,longitude
0,demo-vehicle-01,trip-001,0.0,false,0.0,10.7769,106.7009
```

Required fields: `timestamp_ms`, `vehicle_id`, `trip_id`, `ego_speed_mps`, `brake_active`, `steering_angle_deg`.

`latitude` and `longitude` are optional for the core TTC pipeline, but required to create a dashboard route heatmap. Telemetry timestamps must be monotonic and aligned to the road-facing video clock.
