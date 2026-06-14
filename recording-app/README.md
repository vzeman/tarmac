# RoadSurvey Recorder

Flutter mobile recorder for the Tarmac road-survey workflow. This implementation covers milestones 1 and 2 from `SPEC.md`: project scaffold, settings persistence, permission wiring, continuous video capture, GPS/IMU telemetry streams, and JSON/GPX sidecar output.

## Run

Use the Flutter SDK requested by the repo task:

```sh
export PATH="/Users/viktorzeman/development/flutter/bin:$PATH"
flutter pub get
flutter run
```

The app targets iOS 16+ and Android API 29+. Android manifests are wired, but this machine does not have an Android SDK available for a local Android build.

## Implemented

- Flutter app scaffold under `recording-app/` with package `roadsurvey_recorder` and org `com.qualityunit`.
- Dependencies pinned in `pubspec.yaml`: `camera`, `geolocator`, `sensors_plus`, `permission_handler`, `path_provider`, `shared_preferences`, `flutter_map`, `latlong2`, `wakelock_plus`, and `intl`.
- Settings model and `shared_preferences` persistence for all M1 settings.
- iOS and Android camera/location/motion permission metadata.
- Permission rationale/request flow before recording.
- Continuous video camera preview and start/stop recording to app internal documents storage.
- GPS stream at highest practical plugin rate with UTC and PTS timestamps.
- Accelerometer and gyroscope sampling at a 10 ms requested interval with UTC and PTS timestamps.
- Sidecar writer streams GPS/IMU to temporary NDJSON files while recording, flushes periodically, and finalizes:
  - `<session>_seg001.track.json`
  - `<session>_seg001.gpx`
  - `<session>_seg001.<camera-extension>`
- Sessions index, sessions list, and track detail map.

## Sidecar Shape

The JSON sidecar uses `schema: "roadsurvey.track.v1"` and contains:

- `session`: app/device/camera/settings snapshot, segment index, start/end UTC, requested and actual mode.
- `frames[]`: continuous-video frame rows derived from configured FPS and duration, with interpolated GPS where available.
- `gps[]`: raw GPS samples.
- `imu[]`: raw combined accelerometer/gyroscope samples.

## Stubbed For Later Milestones

Explicit TODO stubs live in `lib/services/future_milestone_stubs.dart` and the session metadata notes:

- SPEC M3: adaptive distance/FPS capture.
- SPEC M3: stationary auto-pause with GPS and accelerometer fusion.
- SPEC M4: external USB-C/SAF storage routing.
- SPEC M4: max-size segment auto-split.
- SPEC M7: background recording, foreground service details, thermal/battery policy.
- SPEC M6: Tarmac `survey --gps-sidecar` ingestion.
