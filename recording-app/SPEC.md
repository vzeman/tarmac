# RoadSurvey Recorder — Mobile App Specification

Cross-platform (iOS + Android) field-recording app that captures **geo-referenced road-surface video/frames** for the Tarmac analysis pipeline. Its job is to produce, for every captured frame, an **accurate timestamp + GPS position + motion data** — solving the gap where stock phone video stores only a single start location (no continuous GPS track).

## 1. Goals & non-goals
**Goals**
- Record road footage hands-free from a vehicle/motorbike mount.
- Attach a **continuous, per-frame GPS + IMU + timestamp sidecar** synchronized to the imagery.
- **Clever recording**: auto-pause when stationary; **adaptive capture rate tied to speed** (constant spatial sampling, not constant time sampling).
- Write to **external storage (USB-C SSD)** when connected; otherwise internal.
- **Auto-split** output into segments ≤ a configurable max size (default 10 GB).
- Output a format the Tarmac `survey` pipeline ingests directly (accurate positions → no IMU dead-reckoning).

**Non-goals**: on-device ML/crack detection (Tarmac does that server-side); cloud account system (export is file-based / share-sheet for v1).

## 2. Platform & stack
- **Flutter (Dart)**, single codebase. Targets: **iOS 16+** (iPhone 15+ for USB-C external SSD), **Android 10+ / API 29+**.
- Core packages (candidates): `camera` or `camerawesome` (capture), `geolocator` (GPS, speed, heading), `sensors_plus` (accel/gyro), `permission_handler`, `path_provider`, `shared_preferences` (settings), `flutter_map` + offline-capable tiles (live track), `wakelock_plus` (keep screen/CPU awake).
- **Platform channels (native Kotlin/Swift)** required for: (a) external USB storage access, (b) high-precision capture timestamps / frame PTS, (c) GPS-time discipline. These are the parts Flutter plugins don't cover and must be written per-platform.

## 3. Capture model
Two selectable modes; **Adaptive Frame Capture is the default & recommended** for surveys.

### 3A. Adaptive Frame Capture (default)
- Capture individual high-resolution frames at a cadence driven by **speed**, targeting a constant **frame spacing in meters** (setting, default **3 m**).
- `capture_interval_seconds = frame_spacing_m / max(speed_mps, epsilon)`, i.e. effective fps = `speed_mps / frame_spacing_m`, clamped to **`max_fps`** (setting, default 8) and **`min_fps`** when moving slowly.
- **Stationary → pause**: if speed < `pause_speed_kmh` (default 2) for > `pause_debounce_s` (default 3), stop capturing frames (and stop writing) until movement resumes. Debounce prevents flapping at lights/junctions. Optional "heartbeat" frame every N seconds while paused (off by default).
- Frames are written as a **VFR (variable-frame-rate) video** with exact per-frame presentation timestamps **and/or** a frame sequence + manifest (see §4). VFR video keeps files small and frames sharp; the sidecar carries true timing.
- Rationale: faster vehicle → more frames/sec to keep ~constant ground spacing; slow/stopped → few/none. This gives the analysis pipeline uniform spatial coverage and avoids redundant frames.

### 3B. Continuous Video (alternative)
- Standard fixed-fps video (e.g. 30 fps) + the full per-frame sidecar. Adaptive/distance sampling is then done downstream by Tarmac. Larger files; use when uninterrupted footage is desired.

### Motion detection
- Primary: GPS speed. Secondary/faster: **accelerometer** magnitude change (detects start/stop sooner than GPS, useful in tunnels/GPS gaps). Fuse both; accelerometer triggers "moving" instantly, GPS confirms.

## 4. Geospatial & temporal data (the core value)
Captured continuously while recording, at the highest sustainable rate:
- **GPS**: latitude, longitude, altitude, horizontal/vertical accuracy, speed, heading, fix timestamp (UTC, GPS-disciplined), satellite count if available. Target ≥ 1 Hz (10 Hz where the device supports it).
- **IMU**: accelerometer (x,y,z, m/s²), gyroscope (x,y,z, rad/s), at 50–120 Hz.
- **Timestamps**: every frame gets (a) a monotonic-clock PTS for ordering, and (b) an **absolute UTC time derived from the GPS clock** for cross-device accuracy. Frame PTS ↔ GPS samples are aligned so each frame can be assigned an interpolated lat/lon/speed.

### Sidecar format (per segment) — consumed by Tarmac
Write **both** a machine JSON and a standard GPX:
- `*.track.json`:
  - `session`: app version, device model/OS, camera resolution & intrinsics (focal length, sensor size if available), mode, settings snapshot, segment index, start/end UTC.
  - `frames[]`: `{frame_index, pts_ms, utc_ms, lat, lon, alt_m, gps_accuracy_m, speed_mps, heading_deg}`.
  - `imu[]`: `{utc_ms, ax, ay, az, gx, gy, gz}` (decimated or full).
- `*.gpx`: standard track for quick viewing in any GIS/map tool.
- **Tarmac integration**: add `tarmac survey --gps-sidecar <track.json>` so the pipeline reads true per-frame GPS instead of dead-reckoning (closes the loop with the current IMU-only limitation).

## 5. Storage
- **Location setting**: Auto (prefer external if connected) / Internal / External.
- **External**: iOS — security-scoped bookmark to a USB-C volume via the document picker (native Swift channel); detect connect/disconnect. Android — Storage Access Framework (SAF) tree URI or USB-OTG mass storage; detect via `UsbManager`.
- **Auto-split**: finalize current segment and start a new one when it reaches **`max_segment_gb`** (setting, default **10 GB**; range 1–50). Each segment is independently valid (own moov atom / finalized container) + its own sidecar, so a crash loses at most the in-progress segment.
- **Naming**: `<session_id>/<session_id>_seg<NNN>.{mov|mp4}` + `<...>_seg<NNN>.track.json` + `.gpx`.
- **Storage guard**: stop & warn when free space < threshold; never fill the volume.

## 6. UI / UX
- **Record screen**: live camera preview, big Start/Stop, live mini-map of the track, current speed, captured-frame count, current segment size, free storage, recording/paused state, GPS-fix quality indicator.
- **Sessions screen**: list past sessions (thumbnail, date, distance, duration, #frames, #segments, size); open detail with map of the track + export.
- **Settings**: frame spacing (m), max/min fps, pause speed & debounce, capture mode (Adaptive/Continuous), resolution & codec, max segment size (GB), storage location, sidecar options, keep-screen-on, units.
- **Export screen**: copy/share segments + sidecars (to Files/SSD, AirDrop, or HTTP upload to a Tarmac ingest endpoint — v2).

## 7. Permissions, power, reliability
- Permissions: camera, location (**Always**, for background), motion/sensors, storage. Clear rationale prompts.
- **Background recording**: continue when screen locked/app backgrounded (iOS background modes: location + (limited) capture caveats; Android foreground service with notification). Document iOS background-camera limitations honestly.
- **Power/thermal**: continuous camera+GPS+sensors is heavy — expose resolution/fps caps, show battery/thermal state, and degrade gracefully (lower fps) on thermal pressure.
- **Reliability**: crash-safe segment finalization; periodic sidecar flush (don't hold all telemetry in memory); resume/finalize a half-written segment on relaunch; handle GPS loss (mark frames `gps_accuracy=null`, optionally fill via IMU but flag).

## 8. Non-functional targets
- Works fully **offline** (GPS + local storage; map tiles cached/optional).
- Timestamp accuracy: frame↔GPS alignment within one GPS sample period.
- No frame without a telemetry record (interpolated if between GPS fixes).
- Battery: target ≥ 2 h continuous on a typical phone at default settings.

## 9. Development milestones (for codex)
1. **Scaffold**: Flutter project in `recording-app/`, package structure, CI-less local run, permissions wiring, settings store. Builds on iOS + Android.
2. **Capture + telemetry core**: camera preview/record, GPS + IMU streaming, per-frame timestamping, sidecar writer (JSON+GPX). Continuous-video mode first.
3. **Clever recording**: adaptive capture rate, stationary auto-pause (GPS+accel fusion), debounce.
4. **Storage**: segment auto-split at max size; internal storage; then external (USB-C/SAF) via native channels.
5. **UI**: record screen (preview/map/HUD), sessions list, settings, export/share.
6. **Tarmac integration**: finalize sidecar schema; add `tarmac survey --gps-sidecar` ingestion on the Python side; round-trip test (record short clip → sidecar → Tarmac survey produces an accurate map).
7. **Hardening**: background mode, thermal/battery, crash-safe finalize, field test.

## 10. Open decisions / trade-offs (flagged)
- **Adaptive-fps VFR video vs frame-sequence**: true VFR encoding is fiddly cross-platform; the robust fallback is a frame sequence + manifest that Tarmac assembles. Recommend implementing frame-sequence first, VFR video as enhancement.
- **iOS external-SSD direct write & background camera** are the highest-risk native pieces; may need an iOS-first native spike before committing the cross-platform API.
- **GPS rate**: 10 Hz isn't available on all devices; default 1 Hz with interpolation, opt into higher where supported.
