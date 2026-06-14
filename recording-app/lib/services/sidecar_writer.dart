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
    required this.startUtc,
  });

  final String sessionId;
  final Directory sessionDirectory;
  final AppSettings settings;
  final CaptureMode actualMode;
  final DateTime startUtc;

  late final File _gpsTempFile;
  late final File _imuTempFile;
  IOSink? _gpsSink;
  IOSink? _imuSink;

  int gpsSampleCount = 0;
  int imuSampleCount = 0;
  GpsSample? firstGps;
  GpsSample? lastGps;

  String get _segmentBase => '${sessionId}_seg001';

  File get sidecarFile =>
      File('${sessionDirectory.path}/$_segmentBase.track.json');

  File get gpxFile => File('${sessionDirectory.path}/$_segmentBase.gpx');

  static Future<SidecarWriter> open({
    required String sessionId,
    required Directory sessionDirectory,
    required AppSettings settings,
    required CaptureMode actualMode,
    required DateTime startUtc,
  }) async {
    final writer = SidecarWriter._(
      sessionId: sessionId,
      sessionDirectory: sessionDirectory,
      settings: settings,
      actualMode: actualMode,
      startUtc: startUtc,
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
    sink.writeln(jsonEncode(sample.toJson()));
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
    sink.writeln(jsonEncode(sample.toJson()));
    if (imuSampleCount % 240 == 0) {
      unawaited(sink.flush());
    }
  }

  Future<SessionSummary> finalize({
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

    final totalBytes = await _directorySize(sessionDirectory);
    final start = gpsSamples.isNotEmpty ? gpsSamples.first : firstGps;
    final end = gpsSamples.isNotEmpty ? gpsSamples.last : lastGps;
    return SessionSummary(
      id: sessionId,
      directoryPath: sessionDirectory.path,
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
      mode: actualMode.name,
      startLat: start?.lat,
      startLon: start?.lon,
      endLat: end?.lat,
      endLon: end?.lon,
    );
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
  }

  Future<void> _closeSinks() async {
    final gps = _gpsSink;
    final imu = _imuSink;
    _gpsSink = null;
    _imuSink = null;
    if (gps != null) {
      await gps.flush();
      await gps.close();
    }
    if (imu != null) {
      await imu.flush();
      await imu.close();
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
      'segment_index': 1,
      'start_utc': startUtc.toIso8601String(),
      'end_utc': endUtc.toIso8601String(),
      'duration_ms': durationMs,
      'video_file': videoFile.uri.pathSegments.isEmpty
          ? videoFile.path
          : videoFile.uri.pathSegments.last,
      'notes': {
        'adaptive_capture': 'TODO SPEC M3',
        'stationary_auto_pause': 'TODO SPEC M3',
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

  Future<int> _directorySize(Directory directory) async {
    var total = 0;
    await for (final entity in directory.list(recursive: true)) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  Future<void> _deleteTempFiles() async {
    for (final file in [_gpsTempFile, _imuTempFile]) {
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
