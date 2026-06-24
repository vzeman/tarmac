import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import '../settings/app_settings.dart';
import 'camera_service.dart';
import 'location_service.dart';
import 'lidar_service.dart';
import 'motion_service.dart';
import 'session_repository.dart';
import 'sidecar_writer.dart';
import 'storage_service.dart';
import 'timestamp_service.dart';

class CaptureSessionController extends ChangeNotifier {
  CaptureSessionController({
    required this.settings,
    required this.sessionRepository,
  });

  AppSettings settings;
  final SessionRepository sessionRepository;
  final CameraService cameraService = CameraService();
  final LocationService locationService = LocationService();
  final MotionService motionService = MotionService();
  final LidarService lidarService = LidarService();

  SidecarWriter? _writer;
  StorageTarget? _storageTarget;
  RecordingClock? _clock;
  StreamSubscription<GpsSample>? _gpsSubscription;
  StreamSubscription<ImuSample>? _imuSubscription;
  StreamSubscription<LidarFrame>? _lidarSubscription;
  Timer? _ticker;

  String? _sessionId;
  Directory? _sessionDirectory;
  DateTime? _sessionStartUtc;
  DateTime? _segmentStartUtc;
  int? _segmentStartPtsMs;
  final List<SessionSegment> _segments = [];
  final _MotionVarianceWindow _motionWindow = _MotionVarianceWindow();
  DateTime? _stationarySinceUtc;
  bool _segmentTransitioning = false;
  int _completedDurationMs = 0;
  int _completedFrameCount = 0;

  bool initializingCamera = false;
  bool isRecording = false;
  bool isStopping = false;
  bool isAutoPaused = false;
  String? autoPauseReason;
  String? errorMessage;
  String? warningMessage;
  Duration elapsed = Duration.zero;
  int gpsSamples = 0;
  int imuSamples = 0;
  double speedMps = 0;
  double? gpsAccuracyM;
  double motionVariance = 0;
  DateTime? lastGpsFixUtc;
  final List<TrackPoint> track = [];

  CameraController? get cameraController => cameraService.controller;

  bool get isActivelyRecording =>
      isRecording && !isAutoPaused && _writer != null && !isStopping;

  int get segmentCount => _segments.length + (_writer == null ? 0 : 1);

  int get estimatedFrameCount {
    final activeSegmentFrames = _writer == null
        ? 0
        : ((_currentSegmentDurationMs / 1000) * settings.effectiveContinuousFps)
              .floor();
    return _completedFrameCount + activeSegmentFrames;
  }

  int get _activeDurationMs {
    return _completedDurationMs +
        (_writer == null ? 0 : _currentSegmentDurationMs);
  }

  int get _currentSegmentDurationMs {
    final startPtsMs = _segmentStartPtsMs;
    final clock = _clock;
    if (startPtsMs == null || clock == null) {
      return 0;
    }
    return math.max(0, clock.elapsedMs - startPtsMs);
  }

  Future<void> initializeCamera() async {
    initializingCamera = true;
    errorMessage = null;
    notifyListeners();
    try {
      await cameraService.initialize(settings);
    } on CameraException catch (error) {
      errorMessage = error.description ?? error.code;
    } on Exception catch (error) {
      errorMessage = error.toString();
    } finally {
      initializingCamera = false;
      notifyListeners();
    }
  }

  Future<void> updateSettings(AppSettings next) async {
    settings = next;
    if (!isRecording) {
      await initializeCamera();
      return;
    }
    if (!settings.autoPauseEnabled && isAutoPaused) {
      unawaited(_resumeFromAutoPause(reason: 'auto-pause disabled'));
    }
    notifyListeners();
  }

  Future<SessionSummary?> start({
    StorageTargetType? storageTargetOverride,
  }) async {
    if (isRecording || isStopping) {
      return null;
    }
    _resetLiveSessionState();
    errorMessage = null;
    warningMessage = settings.captureMode == CaptureMode.adaptive
        ? 'Adaptive mode records continuous video; spatial sampling is downstream.'
        : null;
    notifyListeners();

    try {
      await locationService.ensureServiceEnabled();
      await cameraService.initialize(settings);
      _storageTarget = await sessionRepository.storageService.activeTarget(
        settings,
        forceInternal: storageTargetOverride == StorageTargetType.internal,
      );

      final startFix = await locationService.currentBestFix();
      final startUtc = startFix?.timestamp.toUtc() ?? DateTime.now().toUtc();
      final sessionId = sessionRepository.createSessionId(startUtc);
      final sessionDirectory = await sessionRepository.createSessionDirectory(
        sessionId,
      );

      _sessionId = sessionId;
      _sessionDirectory = sessionDirectory;
      _sessionStartUtc = startUtc;
      _clock = RecordingClock(startUtc)..start();
      _attachTelemetryListeners();

      await _startNewSegment(index: 1, startUtc: startUtc, startPtsMs: 0);
      await locationService.start(_clock!);
      await motionService.start(_clock!);
      await lidarService.start(_clock!);

      if (settings.keepScreenOn) {
        await WakelockPlus.enable();
      }
      isRecording = true;
      isAutoPaused = false;
      _ticker = Timer.periodic(const Duration(milliseconds: 500), (_) {
        elapsed = Duration(milliseconds: _activeDurationMs);
        notifyListeners();
      });
      notifyListeners();
      return null;
    } on Exception catch (error) {
      errorMessage = error.toString();
      await _cleanupAfterFailure();
      notifyListeners();
      return null;
    }
  }

  Future<SessionSummary?> stop() async {
    if (!isRecording || isStopping) {
      return null;
    }
    isStopping = true;
    notifyListeners();

    try {
      if (_writer != null) {
        await _finalizeActiveSegment();
      }
      _clock?.stop();
      await _stopTelemetry();
      await WakelockPlus.disable();

      if (_segments.isEmpty) {
        throw StateError('No finalized video segments were captured.');
      }

      var summary = await _buildSessionSummary();
      final manifest = await SidecarWriter.writeSessionManifest(
        summary: summary,
        settings: settings,
        actualMode: CaptureMode.continuous,
      );
      final totalBytes = await sessionRepository.storageService.directorySize(
        _sessionDirectory!,
      );
      summary = summary.copyWith(
        manifestPath: manifest.path,
        totalBytes: totalBytes,
      );

      final finalizedSummary = await _finalizeStorageTarget(summary);
      await sessionRepository.saveSummary(finalizedSummary);
      _resetAfterStop();
      notifyListeners();
      return finalizedSummary;
    } on Exception catch (error) {
      errorMessage = error.toString();
      isStopping = false;
      notifyListeners();
      return null;
    }
  }

  void _attachTelemetryListeners() {
    _gpsSubscription = locationService.samples.listen((sample) {
      gpsSamples += 1;
      speedMps = sample.speedMps ?? 0;
      gpsAccuracyM = sample.horizontalAccuracyM;
      lastGpsFixUtc = DateTime.fromMillisecondsSinceEpoch(
        sample.fixUtcMs,
        isUtc: true,
      );
      track.add(
        TrackPoint(lat: sample.lat, lon: sample.lon, utcMs: sample.utcMs),
      );
      _writer?.addGps(sample);
      _evaluateAutoPause(
        DateTime.fromMillisecondsSinceEpoch(sample.utcMs, isUtc: true),
      );
      notifyListeners();
    });

    _imuSubscription = motionService.samples.listen((sample) {
      imuSamples += 1;
      motionVariance = _motionWindow.add(sample);
      _writer?.addImu(sample);
      if (isAutoPaused &&
          settings.autoPauseEnabled &&
          motionVariance >= settings.resumeMotionVarianceThreshold) {
        unawaited(
          _resumeFromAutoPause(
            reason: 'accelerometer motion',
            triggerUtc: DateTime.fromMillisecondsSinceEpoch(
              sample.utcMs,
              isUtc: true,
            ),
          ),
        );
      } else {
        _evaluateAutoPause(
          DateTime.fromMillisecondsSinceEpoch(sample.utcMs, isUtc: true),
        );
      }
      if (imuSamples % 30 == 0 || isAutoPaused) {
        notifyListeners();
      }
    });

    _lidarSubscription = lidarService.frames.listen((frame) {
      _writer?.addLidar(frame);
    });
  }

  Future<void> _startNewSegment({
    required int index,
    required DateTime startUtc,
    required int startPtsMs,
  }) async {
    final sessionId = _sessionId;
    final sessionDirectory = _sessionDirectory;
    if (sessionId == null || sessionDirectory == null) {
      throw StateError('No active session directory.');
    }
    final writer = await SidecarWriter.open(
      sessionId: sessionId,
      sessionDirectory: sessionDirectory,
      settings: settings,
      actualMode: CaptureMode.continuous,
      segmentIndex: index,
      startUtc: startUtc,
      segmentStartPtsMs: startPtsMs,
    );
    _writer = writer;
    _segmentStartUtc = startUtc;
    _segmentStartPtsMs = startPtsMs;
    try {
      await cameraService.prepareRecording();
      await cameraService.startRecording();
    } on Exception {
      if (_writer == writer) {
        _writer = null;
      }
      await writer.discard();
      _segmentStartUtc = null;
      _segmentStartPtsMs = null;
      rethrow;
    }
  }

  Future<void> _autoPause() async {
    if (_segmentTransitioning ||
        isStopping ||
        !isRecording ||
        isAutoPaused ||
        _writer == null) {
      return;
    }
    _segmentTransitioning = true;
    try {
      await _finalizeActiveSegment();
      isAutoPaused = true;
      autoPauseReason = 'stationary';
      _stationarySinceUtc = null;
    } on Exception catch (error) {
      errorMessage = 'Auto-pause failed: $error';
      if (_writer == null) {
        isAutoPaused = true;
        autoPauseReason = 'stationary';
      }
    } finally {
      _segmentTransitioning = false;
      elapsed = Duration(milliseconds: _activeDurationMs);
      notifyListeners();
    }
  }

  Future<void> _resumeFromAutoPause({
    required String reason,
    DateTime? triggerUtc,
  }) async {
    if (_segmentTransitioning ||
        isStopping ||
        !isRecording ||
        !isAutoPaused ||
        _writer != null) {
      return;
    }
    final clock = _clock;
    if (clock == null) {
      return;
    }
    _segmentTransitioning = true;
    try {
      final startPtsMs = clock.elapsedMs;
      await _startNewSegment(
        index: _segments.length + 1,
        startUtc: triggerUtc ?? clock.nowUtc,
        startPtsMs: startPtsMs,
      );
      isAutoPaused = false;
      autoPauseReason = null;
      _stationarySinceUtc = null;
    } on Exception catch (error) {
      errorMessage = 'Auto-resume failed after $reason: $error';
      isAutoPaused = true;
    } finally {
      _segmentTransitioning = false;
      elapsed = Duration(milliseconds: _activeDurationMs);
      notifyListeners();
    }
  }

  Future<void> _finalizeActiveSegment() async {
    final writer = _writer;
    final startUtc = _segmentStartUtc;
    final startPtsMs = _segmentStartPtsMs;
    final sessionDirectory = _sessionDirectory;
    final sessionId = _sessionId;
    if (writer == null ||
        startUtc == null ||
        startPtsMs == null ||
        sessionDirectory == null ||
        sessionId == null) {
      return;
    }

    final capturedFile = await cameraService.stopRecording();
    final clock = _clock;
    final endPtsMs = clock?.elapsedMs ?? startPtsMs;
    final endUtc = clock?.nowUtc ?? DateTime.now().toUtc();
    final durationMs = math.max(0, endPtsMs - startPtsMs);
    _writer = null;
    _segmentStartUtc = null;
    _segmentStartPtsMs = null;

    final videoFile = await cameraService.persistRecording(
      capturedFile: capturedFile,
      sessionDirectory: sessionDirectory,
      sessionId: sessionId,
      segmentIndex: writer.segmentIndex,
    );
    final segment = await writer.finalize(
      endUtc: endUtc,
      durationMs: durationMs,
      videoFile: videoFile,
    );
    _segments.add(segment);
    _completedDurationMs += segment.durationMs;
    _completedFrameCount += segment.frameCount;
    elapsed = Duration(milliseconds: _activeDurationMs);
  }

  void _evaluateAutoPause(DateTime nowUtc) {
    if (!settings.autoPauseEnabled ||
        !isRecording ||
        isStopping ||
        isAutoPaused ||
        _segmentTransitioning ||
        _writer == null) {
      _stationarySinceUtc = null;
      return;
    }

    final gpsStationary = speedMps * 3.6 < settings.pauseSpeedKmh;
    final motionStationary =
        _motionWindow.hasEnoughSamples &&
        motionVariance <= settings.pauseMotionVarianceThreshold;
    if (!gpsStationary || !motionStationary) {
      _stationarySinceUtc = null;
      return;
    }

    _stationarySinceUtc ??= nowUtc;
    if (nowUtc.difference(_stationarySinceUtc!).inMilliseconds >
        settings.pauseDebounceS * 1000) {
      unawaited(_autoPause());
    }
  }

  Future<SessionSummary> _buildSessionSummary() async {
    final sessionDirectory = _sessionDirectory;
    final sessionId = _sessionId;
    if (sessionDirectory == null || sessionId == null || _segments.isEmpty) {
      throw StateError('No finalized session segments.');
    }

    final first = _segments.first;
    final last = _segments.last;
    final durationMs = _segments.fold<int>(
      0,
      (sum, segment) => sum + segment.durationMs,
    );
    final frameCount = _segments.fold<int>(
      0,
      (sum, segment) => sum + segment.frameCount,
    );
    final gpsSampleCount = _segments.fold<int>(
      0,
      (sum, segment) => sum + segment.gpsSampleCount,
    );
    final imuSampleCount = _segments.fold<int>(
      0,
      (sum, segment) => sum + segment.imuSampleCount,
    );

    return SessionSummary(
      id: sessionId,
      directoryPath: sessionDirectory.path,
      videoPath: first.videoPath,
      sidecarPath: first.sidecarPath,
      gpxPath: first.gpxPath,
      startedAtUtc: _sessionStartUtc ?? first.startedAtUtc,
      endedAtUtc: last.endedAtUtc,
      durationMs: durationMs,
      frameCount: frameCount,
      gpsSampleCount: gpsSampleCount,
      imuSampleCount: imuSampleCount,
      totalBytes: await sessionRepository.storageService.directorySize(
        sessionDirectory,
      ),
      mode: CaptureMode.continuous.name,
      segments: List<SessionSegment>.unmodifiable(_segments),
      startLat: first.startLat,
      startLon: first.startLon,
      endLat: last.endLat,
      endLon: last.endLon,
    );
  }

  Future<void> _cleanupAfterFailure() async {
    await _stopTelemetry();
    await WakelockPlus.disable();
    await _stopCameraIfRecording();
    await _writer?.discard();
    _resetAfterStop();
  }

  Future<void> _stopCameraIfRecording() async {
    final controller = cameraService.controller;
    if (controller?.value.isRecordingVideo != true) {
      return;
    }
    try {
      final captured = await cameraService.stopRecording();
      final temp = File(captured.path);
      if (await temp.exists()) {
        await temp.delete();
      }
    } on Exception {
      return;
    }
  }

  Future<SessionSummary> _finalizeStorageTarget(SessionSummary summary) async {
    final target =
        _storageTarget ??
        await sessionRepository.storageService.activeTarget(settings);
    try {
      return await sessionRepository.storageService.finalizeToTarget(
        summary: summary,
        target: target,
      );
    } on Exception {
      if (!target.isExternal) {
        rethrow;
      }
      warningMessage = 'External storage became unavailable; saved internally.';
      return summary.copyWith(storageLocation: 'internal');
    }
  }

  Future<void> _stopTelemetry() async {
    _ticker?.cancel();
    _ticker = null;
    await _gpsSubscription?.cancel();
    await _imuSubscription?.cancel();
    _gpsSubscription = null;
    _imuSubscription = null;
    await _lidarSubscription?.cancel();
    _lidarSubscription = null;
    await locationService.stop();
    await motionService.stop();
    await lidarService.stop();
  }

  void _resetLiveSessionState() {
    _writer = null;
    _storageTarget = null;
    _clock = null;
    _sessionId = null;
    _sessionDirectory = null;
    _sessionStartUtc = null;
    _segmentStartUtc = null;
    _segmentStartPtsMs = null;
    _segments.clear();
    _motionWindow.clear();
    _stationarySinceUtc = null;
    _segmentTransitioning = false;
    _completedDurationMs = 0;
    _completedFrameCount = 0;
    isRecording = false;
    isStopping = false;
    isAutoPaused = false;
    autoPauseReason = null;
    elapsed = Duration.zero;
    gpsSamples = 0;
    imuSamples = 0;
    speedMps = 0;
    gpsAccuracyM = null;
    motionVariance = 0;
    lastGpsFixUtc = null;
    track.clear();
  }

  void _resetAfterStop() {
    _resetLiveSessionState();
    _storageTarget = null;
  }

  @override
  void dispose() {
    unawaited(_stopTelemetry());
    unawaited(WakelockPlus.disable());
    unawaited(_writer?.discard());
    unawaited(cameraService.dispose());
    lidarService.dispose();
    super.dispose();
  }
}

class _MotionVarianceWindow {
  static const _windowMs = 900;
  static const _minSamples = 8;

  final List<_MotionPoint> _samples = [];

  bool get hasEnoughSamples => _samples.length >= _minSamples;

  double add(ImuSample sample) {
    final magnitude = math.sqrt(
      (sample.ax * sample.ax) +
          (sample.ay * sample.ay) +
          (sample.az * sample.az),
    );
    _samples.add(_MotionPoint(sample.utcMs, magnitude));
    final cutoff = sample.utcMs - _windowMs;
    _samples.removeWhere((point) => point.utcMs < cutoff);
    if (_samples.length < 2) {
      return 0;
    }
    final mean =
        _samples.fold<double>(0, (sum, point) => sum + point.magnitude) /
        _samples.length;
    final variance =
        _samples.fold<double>(0, (sum, point) {
          final delta = point.magnitude - mean;
          return sum + (delta * delta);
        }) /
        (_samples.length - 1);
    return variance;
  }

  void clear() {
    _samples.clear();
  }
}

class _MotionPoint {
  const _MotionPoint(this.utcMs, this.magnitude);

  final int utcMs;
  final double magnitude;
}
