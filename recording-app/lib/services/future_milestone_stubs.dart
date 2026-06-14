class FutureMilestoneStubs {
  // TODO SPEC M3: Replace continuous-video frame derivation with adaptive
  // distance-based capture and exact per-frame PTS once native timing lands.
  Future<void> configureAdaptiveDistanceCapture() async {}

  // TODO SPEC M3: Fuse GPS speed and accelerometer movement for stationary
  // auto-pause with debounce.
  Future<void> configureStationaryAutoPause() async {}

  // TODO SPEC M4: Route segment writers to iOS security-scoped USB volumes or
  // Android SAF tree URIs, with internal storage fallback.
  Future<void> configureExternalStorage() async {}

  // TODO SPEC M4: Finalize and roll to the next valid segment at max size.
  Future<void> configureSegmentAutoSplit() async {}

  // TODO SPEC M7: Add foreground service/background mode integration and
  // thermal/battery degradation policy.
  Future<void> configureBackgroundAndThermalPolicy() async {}

  // TODO SPEC M6: Add Python-side `tarmac survey --gps-sidecar` ingestion.
  Future<void> configureTarmacIngestionContract() async {}
}
