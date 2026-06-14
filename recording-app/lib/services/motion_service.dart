import 'dart:async';

import 'package:sensors_plus/sensors_plus.dart';

import '../models/telemetry.dart';
import 'timestamp_service.dart';

class MotionService {
  final StreamController<ImuSample> _sampleController =
      StreamController<ImuSample>.broadcast();

  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;

  double _ax = 0;
  double _ay = 0;
  double _az = 0;
  double _gx = 0;
  double _gy = 0;
  double _gz = 0;

  Stream<ImuSample> get samples => _sampleController.stream;

  Future<void> start(RecordingClock clock) async {
    await stop();
    _accelerometerSubscription =
        accelerometerEventStream(
          samplingPeriod: const Duration(milliseconds: 10),
        ).listen((event) {
          _ax = event.x;
          _ay = event.y;
          _az = event.z;
          _emit(clock);
        });

    _gyroscopeSubscription =
        gyroscopeEventStream(
          samplingPeriod: const Duration(milliseconds: 10),
        ).listen((event) {
          _gx = event.x;
          _gy = event.y;
          _gz = event.z;
          _emit(clock);
        });
  }

  Future<void> stop() async {
    await _accelerometerSubscription?.cancel();
    await _gyroscopeSubscription?.cancel();
    _accelerometerSubscription = null;
    _gyroscopeSubscription = null;
  }

  void _emit(RecordingClock clock) {
    _sampleController.add(
      ImuSample(
        utcMs: clock.utcMs,
        ptsMs: clock.ptsMs,
        ax: _ax,
        ay: _ay,
        az: _az,
        gx: _gx,
        gy: _gy,
        gz: _gz,
      ),
    );
  }
}
