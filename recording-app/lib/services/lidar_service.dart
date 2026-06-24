import 'dart:async';

import 'package:flutter/services.dart';

import '../models/telemetry.dart';
import 'timestamp_service.dart';

// Wraps the native ARKit LiDAR + visual-odometry plugin.
// On non-LiDAR devices the pose and vert_accel fields are still delivered
// (ARKit runs without depth semantics); depth_f32 / roughness will be absent.
class LidarService {
  static const _method = MethodChannel('roadsurvey_recorder/lidar');
  static const _events = EventChannel('roadsurvey_recorder/lidar/frames');

  final StreamController<LidarFrame> _controller =
      StreamController<LidarFrame>.broadcast();

  StreamSubscription<dynamic>? _nativeSub;

  Stream<LidarFrame> get frames => _controller.stream;

  Future<bool> get isLidarAvailable async {
    try {
      final result = await _method.invokeMethod<bool>('isLidarAvailable');
      return result ?? false;
    } catch (_) {
      return false;
    }
  }

  Future<bool> get isArSupported async {
    try {
      final result = await _method.invokeMethod<bool>('isArSupported');
      return result ?? false;
    } catch (_) {
      return false;
    }
  }

  // captureDepth: false for drive-by mode (pose + accel only, saves battery).
  // captureDepth: true for scan mode (full 32x24 depth map at 10 fps).
  Future<void> start(RecordingClock clock, {bool captureDepth = true}) async {
    await stop();
    await _method.invokeMethod<void>('start', {'captureDepth': captureDepth});
    _nativeSub = _events.receiveBroadcastStream().listen(
      (dynamic raw) {
        if (raw is! Map) return;
        final map = Map<String, dynamic>.from(raw as Map);
        try {
          _controller.add(LidarFrame.fromNative(map));
        } catch (_) {
          // Malformed frame — skip silently.
        }
      },
      onError: (_) {},
    );
  }

  Future<void> stop() async {
    await _nativeSub?.cancel();
    _nativeSub = null;
    try {
      await _method.invokeMethod<void>('stop');
    } catch (_) {}
  }

  void dispose() {
    _nativeSub?.cancel();
    _controller.close();
  }
}
