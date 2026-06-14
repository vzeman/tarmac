import 'dart:async';
import 'dart:io';

import 'package:geolocator/geolocator.dart';

import '../models/telemetry.dart';
import 'timestamp_service.dart';

class LocationService {
  final StreamController<GpsSample> _sampleController =
      StreamController<GpsSample>.broadcast();
  StreamSubscription<Position>? _positionSubscription;

  Position? latestPosition;

  Stream<GpsSample> get samples => _sampleController.stream;

  double get currentSpeedMps => latestPosition?.speed ?? 0;

  Future<void> ensureServiceEnabled() async {
    final enabled = await Geolocator.isLocationServiceEnabled();
    if (!enabled) {
      throw StateError('Location services are disabled.');
    }
  }

  Future<Position?> currentBestFix() async {
    try {
      return await Geolocator.getCurrentPosition(
        locationSettings: _locationSettings(
          timeLimit: const Duration(seconds: 4),
        ),
      );
    } on Exception {
      return Geolocator.getLastKnownPosition();
    }
  }

  Future<void> start(RecordingClock clock) async {
    await stop();
    await ensureServiceEnabled();
    _positionSubscription =
        Geolocator.getPositionStream(
          locationSettings: _locationSettings(),
        ).listen((position) {
          latestPosition = position;
          _sampleController.add(_toSample(position, clock));
        });
  }

  Future<void> stop() async {
    await _positionSubscription?.cancel();
    _positionSubscription = null;
  }

  LocationSettings _locationSettings({Duration? timeLimit}) {
    if (Platform.isAndroid) {
      return AndroidSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0,
        intervalDuration: const Duration(seconds: 1),
        timeLimit: timeLimit,
      );
    }
    if (Platform.isIOS) {
      return AppleSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        activityType: ActivityType.automotiveNavigation,
        allowBackgroundLocationUpdates: true,
        pauseLocationUpdatesAutomatically: false,
        showBackgroundLocationIndicator: true,
        distanceFilter: 0,
        timeLimit: timeLimit,
      );
    }
    return LocationSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 0,
      timeLimit: timeLimit,
    );
  }

  GpsSample _toSample(Position position, RecordingClock clock) {
    return GpsSample(
      utcMs: clock.utcMs,
      ptsMs: clock.ptsMs,
      fixUtcMs: position.timestamp.toUtc().millisecondsSinceEpoch,
      lat: position.latitude,
      lon: position.longitude,
      altM: _positiveOrNull(position.altitude),
      horizontalAccuracyM: _positiveOrNull(position.accuracy),
      verticalAccuracyM: _positiveOrNull(position.altitudeAccuracy),
      speedMps: position.speed.isFinite ? position.speed : null,
      headingDeg: position.heading.isFinite ? position.heading : null,
    );
  }

  double? _positiveOrNull(double value) {
    if (value.isFinite && value > 0) {
      return value;
    }
    return null;
  }
}
