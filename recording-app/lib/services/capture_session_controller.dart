import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import '../settings/app_settings.dart';
import 'camera_service.dart';
import 'location_service.dart';
import 'motion_service.dart';
import 'session_repository.dart';
import 'sidecar_writer.dart';
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

  SidecarWriter? _writer;
  RecordingClock? _clock;
  StreamSubscription<GpsSample>? _gpsSubscription;
  StreamSubscription<ImuSample>? _imuSubscription;
  Timer? _ticker;

  bool initializingCamera = false;
  bool isRecording = false;
  bool isStopping = false;
  String? errorMessage;
  String? warningMessage;
  Duration elapsed = Duration.zero;
  int gpsSamples = 0;
  int imuSamples = 0;
  double speedMps = 0;
  double? gpsAccuracyM;
  DateTime? lastGpsFixUtc;
  final List<TrackPoint> track = [];

  CameraController? get cameraController => cameraService.controller;

  int get estimatedFrameCount {
    return ((elapsed.inMilliseconds / 1000) * settings.effectiveContinuousFps)
        .floor();
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
    } else {
      notifyListeners();
    }
  }

  Future<SessionSummary?> start() async {
    if (isRecording || isStopping) {
      return null;
    }
    errorMessage = null;
    warningMessage = settings.captureMode == CaptureMode.adaptive
        ? 'Adaptive capture is stubbed for SPEC M3; M2 records continuous video.'
        : null;
    track.clear();
    gpsSamples = 0;
    imuSamples = 0;
    elapsed = Duration.zero;
    notifyListeners();

    SidecarWriter? writer;
    try {
      await locationService.ensureServiceEnabled();
      await cameraService.initialize(settings);
      await cameraService.prepareRecording();
      final startFix = await locationService.currentBestFix();
      final startUtc = startFix?.timestamp.toUtc() ?? DateTime.now().toUtc();
      final sessionId = sessionRepository.createSessionId(startUtc);
      final sessionDirectory = await sessionRepository.createSessionDirectory(
        sessionId,
      );
      writer = await SidecarWriter.open(
        sessionId: sessionId,
        sessionDirectory: sessionDirectory,
        settings: settings,
        actualMode: CaptureMode.continuous,
        startUtc: startUtc,
      );
      _writer = writer;
      _clock = RecordingClock(startUtc)..start();

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
        writer?.addGps(sample);
        notifyListeners();
      });
      _imuSubscription = motionService.samples.listen((sample) {
        imuSamples += 1;
        writer?.addImu(sample);
        if (imuSamples % 30 == 0) {
          notifyListeners();
        }
      });

      await locationService.start(_clock!);
      await motionService.start(_clock!);
      await cameraService.startRecording();
      if (settings.keepScreenOn) {
        await WakelockPlus.enable();
      }
      isRecording = true;
      _ticker = Timer.periodic(const Duration(milliseconds: 500), (_) {
        final clock = _clock;
        if (clock != null) {
          elapsed = Duration(milliseconds: clock.elapsedMs);
          notifyListeners();
        }
      });
      notifyListeners();
      return null;
    } on Exception catch (error) {
      errorMessage = error.toString();
      await _cleanupAfterFailure(writer);
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
      final capturedFile = await cameraService.stopRecording();
      final clock = _clock;
      clock?.stop();
      final durationMs = clock?.elapsedMs ?? elapsed.inMilliseconds;
      final endUtc = clock?.nowUtc ?? DateTime.now().toUtc();
      await _stopTelemetry();
      await WakelockPlus.disable();

      final writer = _writer;
      if (writer == null) {
        throw StateError('No active sidecar writer.');
      }
      final videoFile = await cameraService.persistRecording(
        capturedFile: capturedFile,
        sessionDirectory: writer.sessionDirectory,
        sessionId: writer.sessionId,
      );
      final summary = await writer.finalize(
        endUtc: endUtc,
        durationMs: durationMs,
        videoFile: videoFile,
      );
      await sessionRepository.saveSummary(summary);
      _writer = null;
      isRecording = false;
      isStopping = false;
      elapsed = Duration.zero;
      notifyListeners();
      return summary;
    } on Exception catch (error) {
      errorMessage = error.toString();
      isStopping = false;
      notifyListeners();
      return null;
    }
  }

  Future<void> _cleanupAfterFailure(SidecarWriter? writer) async {
    await _stopTelemetry();
    await WakelockPlus.disable();
    await writer?.discard();
    _writer = null;
    _clock = null;
    isRecording = false;
    isStopping = false;
  }

  Future<void> _stopTelemetry() async {
    _ticker?.cancel();
    _ticker = null;
    await _gpsSubscription?.cancel();
    await _imuSubscription?.cancel();
    _gpsSubscription = null;
    _imuSubscription = null;
    await locationService.stop();
    await motionService.stop();
  }

  @override
  void dispose() {
    unawaited(_stopTelemetry());
    unawaited(WakelockPlus.disable());
    unawaited(_writer?.discard());
    unawaited(cameraService.dispose());
    super.dispose();
  }
}
