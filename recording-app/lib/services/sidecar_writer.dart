import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import '../settings/app_settings.dart';

class SidecarWriter {
  SidecarWriter._({
    required this.sessionId,
    required this.sessionDirectory,
    required this.settings,
    required this.actualMode,
    required this.segmentIndex,
    required this.startUtc,
    required this.segmentStartPtsMs,
  });

  final String sessionId;
  final Directory sessionDirectory;
  final AppSettings settings;
  final CaptureMode actualMode;
  final int segmentIndex;
  final DateTime startUtc;
  final int segmentStartPtsMs;

  late final File _gpsTempFile;
  late final File _imuTempFile;
  IOSink? _gpsSink;
  IOSink? _imuSink;
  late final File _lidarTempFile;
  IOSink? _lidarSink;

  int gpsSampleCount = 0;
  int imuSampleCount = 0;
  int lidarFrameCount = 0;
  GpsSample? firstGps;
  GpsSample? lastGps;

  String get _segmentBase =>
      '${sessionId}_seg${segmentIndex.toString().padLeft(3, '0')}';

  File get sidecarFile =>
      File('${sessionDirectory.path}/$_segmentBase.track.json');

  File get gpxFile => File('${sessionDirectory.path}/$_segmentBase.gpx');

  static Future<SidecarWriter> open({
    required String sessionId,
    required Directory sessionDirectory,
    required AppSettings settings,
    required CaptureMode actualMode,
    required int segmentIndex,
    required DateTime startUtc,
    required int segmentStartPtsMs,
  }) async {
    final writer = SidecarWriter._(
      sessionId: sessionId,
      sessionDirectory: sessionDirectory,
      settings: settings,
      actualMode: actualMode,
      segmentIndex: segmentIndex,
      startUtc: startUtc,
      segmentStartPtsMs: segmentStartPtsMs,
    );
    await writer._open();
    return writer;
  }

  void addGps(GpsSample sample) {
    final sink = _gpsSink;
    if (sink == null) {
      return;
    }
    firstGps ??= sample;
    lastGps = sample;
    gpsSampleCount += 1;
    sink.writeln(jsonEncode(_relativeGps(sample).toJson()));
    if (gpsSampleCount % 4 == 0) {
      unawaited(sink.flush());
    }
  }

  void addImu(ImuSample sample) {
    final sink = _imuSink;
    if (sink == null) {
      return;
    }
    imuSampleCount += 1;
    sink.writeln(jsonEncode(_relativeImu(sample).toJson()));
    if (imuSampleCount % 240 == 0) {
      unawaited(sink.flush());
    }
  }

  void addLidar(LidarFrame frame) {
    final sink = _lidarSink;
    if (sink == null) return;
    lidarFrameCount += 1;
    final relative = frame.toJson();
    relative['pts_ms'] = math.max(0, frame.ptsMs - segmentStartPtsMs);
    sink.writeln(jsonEncode(relative));
    if (lidarFrameCount % 100 == 0) {
      unawaited(sink.flush());
    }
  }

  Future<SessionSegment> finalize({
    required DateTime endUtc,
    required int durationMs,
    required File videoFile,
  }) async {
    await _closeSinks();
    final gpsSamples = await _readGpsSamples();
    final frameCount = _frameCount(durationMs);
    await _writeTrackJson(
      gpsSamples: gpsSamples,
      frameCount: frameCount,
      durationMs: durationMs,
      endUtc: endUtc,
      videoFile: videoFile,
    );
    await _writeGpx(gpsSamples);
    await _deleteTempFiles();

    final totalBytes = await _segmentFileBytes(videoFile);
    final start = gpsSamples.isNotEmpty ? gpsSamples.first : firstGps;
    final end = gpsSamples.isNotEmpty ? gpsSamples.last : lastGps;
    return SessionSegment(
      index: segmentIndex,
      videoPath: videoFile.path,
      sidecarPath: sidecarFile.path,
      gpxPath: gpxFile.path,
      startedAtUtc: startUtc,
      endedAtUtc: endUtc,
      durationMs: durationMs,
      frameCount: frameCount,
      gpsSampleCount: gpsSampleCount,
      imuSampleCount: imuSampleCount,
      totalBytes: totalBytes,
      startLat: start?.lat,
      startLon: start?.lon,
      endLat: end?.lat,
      endLon: end?.lon,
    );
  }

  static Future<File> writeSessionManifest({
    required SessionSummary summary,
    required AppSettings settings,
    required CaptureMode actualMode,
  }) async {
    final file = File('${summary.directoryPath}/${summary.id}.session.json');
    final payload = {
      'schema': 'roadsurvey.session.v1',
      'session': {
        'id': summary.id,
        'app': {'name': 'RoadSurvey Recorder', 'version': '1.0.0+1'},
        'device': {
          'os': Platform.operatingSystem,
          'os_version': Platform.operatingSystemVersion,
          'model': null,
        },
        'mode': actualMode.name,
        'requested_capture_mode': settings.captureMode.name,
        'settings': settings.toJson(),
        'started_at_utc': summary.startedAtUtc.toIso8601String(),
        'ended_at_utc': summary.endedAtUtc.toIso8601String(),
        'duration_ms': summary.durationMs,
        'frame_count': summary.frameCount,
        'gps_sample_count': summary.gpsSampleCount,
        'imu_sample_count': summary.imuSampleCount,
        'segment_count': summary.segmentCount,
        'auto_pause': {
          'enabled': settings.autoPauseEnabled,
          'pause_speed_kmh': settings.pauseSpeedKmh,
          'pause_debounce_s': settings.pauseDebounceS,
          'resume_sensitivity': settings.resumeSensitivity,
          'resume_motion_variance_threshold':
              settings.resumeMotionVarianceThreshold,
          'pause_motion_variance_threshold':
              settings.pauseMotionVarianceThreshold,
          'resume_trigger': 'accelerometer_first',
        },
      },
      'segments': [
        for (final segment in summary.effectiveSegments)
          segment.toManifestJson(),
      ],
    };
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(payload),
    );
    return file;
  }

  Future<void> discard() async {
    await _closeSinks();
    await _deleteTempFiles();
  }

  Future<void> _open() async {
    _gpsTempFile = File(
      '${sessionDirectory.path}/$_segmentBase.gps.tmp.ndjson',
    );
    _imuTempFile = File(
      '${sessionDirectory.path}/$_segmentBase.imu.tmp.ndjson',
    );
    _gpsSink = _gpsTempFile.openWrite(mode: FileMode.writeOnlyAppend);
    _imuSink = _imuTempFile.openWrite(mode: FileMode.writeOnlyAppend);
    _lidarTempFile = File(
      '${sessionDirectory.path}/$_segmentBase.lidar.tmp.ndjson',
    );
    _lidarSink = _lidarTempFile.openWrite(mode: FileMode.writeOnlyAppend);
  }

  GpsSample _relativeGps(GpsSample sample) {
    return GpsSample(
      utcMs: sample.utcMs,
      ptsMs: math.max(0, sample.ptsMs - segmentStartPtsMs),
      fixUtcMs: sample.fixUtcMs,
      lat: sample.lat,
      lon: sample.lon,
      altM: sample.altM,
      horizontalAccuracyM: sample.horizontalAccuracyM,
      verticalAccuracyM: sample.verticalAccuracyM,
      speedMps: sample.speedMps,
      headingDeg: sample.headingDeg,
    );
  }

  ImuSample _relativeImu(ImuSample sample) {
    return ImuSample(
      utcMs: sample.utcMs,
      ptsMs: math.max(0, sample.ptsMs - segmentStartPtsMs),
      ax: sample.ax,
      ay: sample.ay,
      az: sample.az,
      gx: sample.gx,
      gy: sample.gy,
      gz: sample.gz,
    );
  }

  Future<void> _closeSinks() async {
    final gps = _gpsSink;
    final imu = _imuSink;
    final lidar = _lidarSink;
    _gpsSink = null;
    _imuSink = null;
    _lidarSink = null;
    if (gps != null) {
      await gps.flush();
      await gps.close();
    }
    if (imu != null) {
      await imu.flush();
      await imu.close();
    }
    if (lidar != null) {
      await lidar.flush();
      await lidar.close();
    }
  }

  Future<List<GpsSample>> _readGpsSamples() async {
    if (!await _gpsTempFile.exists()) {
      return [];
    }
    final samples = <GpsSample>[];
    await for (final line
        in _gpsTempFile
            .openRead()
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) {
        continue;
      }
      try {
        samples.add(
          GpsSample.fromJson(jsonDecode(trimmed) as Map<String, dynamic>),
        );
      } on FormatException {
        continue;
      } on TypeError {
        continue;
      }
    }
    return samples;
  }

  Future<void> _writeTrackJson({
    required List<GpsSample> gpsSamples,
    required int frameCount,
    required int durationMs,
    required DateTime endUtc,
    required File videoFile,
  }) async {
    final sink = sidecarFile.openWrite();
    sink.writeln('{');
    sink.writeln('  "schema": "roadsurvey.track.v1",');
    sink.writeln(
      '  "session": ${jsonEncode(_sessionMetadata(endUtc, durationMs, videoFile))},',
    );
    sink.writeln('  "frames": [');

    var gpsIndex = 0;
    for (var frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      final ptsMs = _framePtsMs(frameIndex);
      final utcMs = startUtc.millisecondsSinceEpoch + ptsMs;
      while (gpsIndex < gpsSamples.length - 2 &&
          gpsSamples[gpsIndex + 1].utcMs < utcMs) {
        gpsIndex += 1;
      }
      final frame = _frameSample(
        frameIndex,
        ptsMs,
        utcMs,
        gpsSamples,
        gpsIndex,
      );
      final comma = frameIndex == frameCount - 1 ? '' : ',';
      sink.writeln('    ${jsonEncode(frame.toJson())}$comma');
    }

    sink.writeln('  ],');
    sink.writeln('  "gps": [');
    await _copyNdjsonArray(sink, _gpsTempFile);
    sink.writeln();
    sink.writeln('  ],');
    sink.writeln('  "imu": [');
    await _copyNdjsonArray(sink, _imuTempFile);
    sink.writeln();
    sink.writeln('  ],');
    sink.writeln('  "lidar": [');
    await _copyNdjsonArray(sink, _lidarTempFile);
    sink.writeln();
    sink.writeln('  ]');
    sink.writeln('}');
    await sink.flush();
    await sink.close();
  }

  Map<String, dynamic> _sessionMetadata(
    DateTime endUtc,
    int durationMs,
    File videoFile,
  ) {
    return {
      'app': {'name': 'RoadSurvey Recorder', 'version': '1.0.0+1'},
      'device': {
        'os': Platform.operatingSystem,
        'os_version': Platform.operatingSystemVersion,
        'model': null,
      },
      'camera': {
        'resolution': settings.resolution.name,
        'requested_fps': settings.effectiveContinuousFps,
        'codec': settings.codec.name,
        'intrinsics': {'focal_length_mm': null, 'sensor_size_mm': null},
      },
      'mode': actualMode.name,
      'requested_capture_mode': settings.captureMode.name,
      'settings': settings.toJson(),
      'auto_pause': {
        'enabled': settings.autoPauseEnabled,
        'pause_speed_kmh': settings.pauseSpeedKmh,
        'pause_debounce_s': settings.pauseDebounceS,
        'resume_sensitivity': settings.resumeSensitivity,
        'resume_trigger': 'accelerometer_first',
      },
      'calibration': {
        'is_set': settings.mountCalibrationSet,
        'mount_height_m': settings.mountHeightM,
        'mount_tilt_deg': settings.mountTiltDeg,
        'lens_profile': settings.lensProfile.name,
      },
      'segment_index': segmentIndex,
      'start_utc': startUtc.toIso8601String(),
      'end_utc': endUtc.toIso8601String(),
      'duration_ms': durationMs,
      'video_file': videoFile.uri.pathSegments.isEmpty
          ? videoFile.path
          : videoFile.uri.pathSegments.last,
      'notes': {
        'adaptive_capture': 'downstream spatial sampling',
        'stationary_auto_pause': 'GPS speed plus linear acceleration variance',
        'segment_auto_split': 'TODO SPEC M4',
        'external_storage': 'TODO SPEC M4',
        'background_recording_thermal': 'TODO SPEC M7',
        'tarmac_ingestion': 'TODO SPEC M6',
      },
    };
  }

  FrameSample _frameSample(
    int frameIndex,
    int ptsMs,
    int utcMs,
    List<GpsSample> gpsSamples,
    int gpsIndex,
  ) {
    final gps = _interpolatedGps(utcMs, gpsSamples, gpsIndex);
    return FrameSample(
      frameIndex: frameIndex,
      ptsMs: ptsMs,
      utcMs: utcMs,
      lat: gps?.lat,
      lon: gps?.lon,
      altM: gps?.altM,
      gpsAccuracyM: gps?.horizontalAccuracyM,
      speedMps: gps?.speedMps,
      headingDeg: gps?.headingDeg,
    );
  }

  GpsSample? _interpolatedGps(
    int utcMs,
    List<GpsSample> samples,
    int gpsIndex,
  ) {
    if (samples.isEmpty) {
      return null;
    }
    if (samples.length == 1 || utcMs <= samples.first.utcMs) {
      return samples.first;
    }
    if (utcMs >= samples.last.utcMs) {
      return samples.last;
    }

    final lower = samples[gpsIndex.clamp(0, samples.length - 1)];
    final upper = samples[(gpsIndex + 1).clamp(0, samples.length - 1)];
    final span = math.max(1, upper.utcMs - lower.utcMs);
    final t = ((utcMs - lower.utcMs) / span).clamp(0.0, 1.0);
    return GpsSample(
      utcMs: utcMs,
      ptsMs: utcMs - startUtc.millisecondsSinceEpoch,
      fixUtcMs: lower.fixUtcMs,
      lat: _lerp(lower.lat, upper.lat, t),
      lon: _lerp(lower.lon, upper.lon, t),
      altM: _lerpNullable(lower.altM, upper.altM, t),
      horizontalAccuracyM: _lerpNullable(
        lower.horizontalAccuracyM,
        upper.horizontalAccuracyM,
        t,
      ),
      verticalAccuracyM: _lerpNullable(
        lower.verticalAccuracyM,
        upper.verticalAccuracyM,
        t,
      ),
      speedMps: _lerpNullable(lower.speedMps, upper.speedMps, t),
      headingDeg: _lerpNullable(lower.headingDeg, upper.headingDeg, t),
    );
  }

  Future<void> _copyNdjsonArray(IOSink sink, File file) async {
    if (!await file.exists()) {
      return;
    }
    var wrote = false;
    await for (final line
        in file
            .openRead()
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) {
        continue;
      }
      if (wrote) {
        sink.writeln(',');
      }
      sink.write('    $trimmed');
      wrote = true;
    }
  }

  Future<void> _writeGpx(List<GpsSample> gpsSamples) async {
    final sink = gpxFile.openWrite();
    sink.writeln('<?xml version="1.0" encoding="UTF-8"?>');
    sink.writeln(
      '<gpx version="1.1" creator="RoadSurvey Recorder" xmlns="http://www.topografix.com/GPX/1/1">',
    );
    sink.writeln('  <trk>');
    sink.writeln('    <name>${_xmlEscape(sessionId)}</name>');
    sink.writeln('    <trkseg>');
    for (final gps in gpsSamples) {
      final time = DateTime.fromMillisecondsSinceEpoch(
        gps.fixUtcMs,
        isUtc: true,
      ).toIso8601String();
      sink.writeln('      <trkpt lat="${gps.lat}" lon="${gps.lon}">');
      if (gps.altM != null) {
        sink.writeln('        <ele>${gps.altM}</ele>');
      }
      sink.writeln('        <time>$time</time>');
      sink.writeln('        <extensions>');
      if (gps.horizontalAccuracyM != null) {
        sink.writeln(
          '          <accuracy_m>${gps.horizontalAccuracyM}</accuracy_m>',
        );
      }
      if (gps.speedMps != null) {
        sink.writeln('          <speed_mps>${gps.speedMps}</speed_mps>');
      }
      if (gps.headingDeg != null) {
        sink.writeln('          <heading_deg>${gps.headingDeg}</heading_deg>');
      }
      sink.writeln('        </extensions>');
      sink.writeln('      </trkpt>');
    }
    sink.writeln('    </trkseg>');
    sink.writeln('  </trk>');
    sink.writeln('</gpx>');
    await sink.flush();
    await sink.close();
  }

  int _frameCount(int durationMs) {
    if (durationMs <= 0) {
      return 0;
    }
    return ((durationMs / 1000) * settings.effectiveContinuousFps).floor() + 1;
  }

  int _framePtsMs(int frameIndex) {
    return ((frameIndex * 1000) / settings.effectiveContinuousFps).round();
  }

  double _lerp(double a, double b, double t) {
    return a + ((b - a) * t);
  }

  double? _lerpNullable(double? a, double? b, double t) {
    if (a == null && b == null) {
      return null;
    }
    if (a == null) {
      return b;
    }
    if (b == null) {
      return a;
    }
    return _lerp(a, b, t);
  }

  Future<int> _segmentFileBytes(File videoFile) async {
    var total = 0;
    for (final file in [videoFile, sidecarFile, gpxFile]) {
      if (await file.exists()) {
        total += await file.length();
      }
    }
    return total;
  }

  Future<void> _deleteTempFiles() async {
    for (final file in [_gpsTempFile, _imuTempFile, _lidarTempFile]) {
      if (await file.exists()) {
        await file.delete();
      }
    }
  }

  String _xmlEscape(String value) {
    return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&apos;');
  }
}
