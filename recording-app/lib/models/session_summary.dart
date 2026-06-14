class SessionSegment {
  const SessionSegment({
    required this.index,
    required this.videoPath,
    required this.sidecarPath,
    required this.gpxPath,
    required this.startedAtUtc,
    required this.endedAtUtc,
    required this.durationMs,
    required this.frameCount,
    required this.gpsSampleCount,
    required this.imuSampleCount,
    this.totalBytes = 0,
    this.startLat,
    this.startLon,
    this.endLat,
    this.endLon,
  });

  final int index;
  final String videoPath;
  final String sidecarPath;
  final String gpxPath;
  final DateTime startedAtUtc;
  final DateTime endedAtUtc;
  final int durationMs;
  final int frameCount;
  final int gpsSampleCount;
  final int imuSampleCount;
  final int totalBytes;
  final double? startLat;
  final double? startLon;
  final double? endLat;
  final double? endLon;

  SessionSegment copyWith({
    String? videoPath,
    String? sidecarPath,
    String? gpxPath,
    int? totalBytes,
  }) {
    return SessionSegment(
      index: index,
      videoPath: videoPath ?? this.videoPath,
      sidecarPath: sidecarPath ?? this.sidecarPath,
      gpxPath: gpxPath ?? this.gpxPath,
      startedAtUtc: startedAtUtc,
      endedAtUtc: endedAtUtc,
      durationMs: durationMs,
      frameCount: frameCount,
      gpsSampleCount: gpsSampleCount,
      imuSampleCount: imuSampleCount,
      totalBytes: totalBytes ?? this.totalBytes,
      startLat: startLat,
      startLon: startLon,
      endLat: endLat,
      endLon: endLon,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'index': index,
      'video_path': videoPath,
      'sidecar_path': sidecarPath,
      'gpx_path': gpxPath,
      'started_at_utc': startedAtUtc.toIso8601String(),
      'ended_at_utc': endedAtUtc.toIso8601String(),
      'duration_ms': durationMs,
      'frame_count': frameCount,
      'gps_sample_count': gpsSampleCount,
      'imu_sample_count': imuSampleCount,
      'total_bytes': totalBytes,
      'start_lat': startLat,
      'start_lon': startLon,
      'end_lat': endLat,
      'end_lon': endLon,
    };
  }

  Map<String, dynamic> toManifestJson() {
    return {
      'index': index,
      'start_utc': startedAtUtc.toIso8601String(),
      'end_utc': endedAtUtc.toIso8601String(),
      'duration_ms': durationMs,
      'frame_count': frameCount,
      'gps_sample_count': gpsSampleCount,
      'imu_sample_count': imuSampleCount,
      'video_file': _basename(videoPath),
      'sidecar_file': _basename(sidecarPath),
      'gpx_file': _basename(gpxPath),
      'start_gps': _gpsJson(startLat, startLon),
      'end_gps': _gpsJson(endLat, endLon),
    };
  }

  factory SessionSegment.fromJson(Map<String, dynamic> json) {
    return SessionSegment(
      index: _readInt(json['index']),
      videoPath: json['video_path']?.toString() ?? '',
      sidecarPath: json['sidecar_path']?.toString() ?? '',
      gpxPath: json['gpx_path']?.toString() ?? '',
      startedAtUtc: _readDate(json['started_at_utc']),
      endedAtUtc: _readDate(json['ended_at_utc']),
      durationMs: _readInt(json['duration_ms']),
      frameCount: _readInt(json['frame_count']),
      gpsSampleCount: _readInt(json['gps_sample_count']),
      imuSampleCount: _readInt(json['imu_sample_count']),
      totalBytes: _readInt(json['total_bytes']),
      startLat: _readDouble(json['start_lat']),
      startLon: _readDouble(json['start_lon']),
      endLat: _readDouble(json['end_lat']),
      endLon: _readDouble(json['end_lon']),
    );
  }
}

class SessionSummary {
  const SessionSummary({
    required this.id,
    required this.directoryPath,
    required this.videoPath,
    required this.sidecarPath,
    required this.gpxPath,
    required this.startedAtUtc,
    required this.endedAtUtc,
    required this.durationMs,
    required this.frameCount,
    required this.gpsSampleCount,
    required this.imuSampleCount,
    required this.totalBytes,
    required this.mode,
    this.manifestPath = '',
    this.segments = const [],
    this.storageLocation = 'internal',
    this.storageAvailable = true,
    this.startLat,
    this.startLon,
    this.endLat,
    this.endLon,
  });

  final String id;
  final String directoryPath;
  final String videoPath;
  final String sidecarPath;
  final String gpxPath;
  final String manifestPath;
  final DateTime startedAtUtc;
  final DateTime endedAtUtc;
  final int durationMs;
  final int frameCount;
  final int gpsSampleCount;
  final int imuSampleCount;
  final int totalBytes;
  final String mode;
  final List<SessionSegment> segments;
  final String storageLocation;
  final bool storageAvailable;
  final double? startLat;
  final double? startLon;
  final double? endLat;
  final double? endLon;

  bool get isExternal => storageLocation == 'external';

  int get segmentCount => effectiveSegments.length;

  List<SessionSegment> get effectiveSegments {
    if (segments.isNotEmpty) {
      return segments;
    }
    if (videoPath.trim().isEmpty &&
        sidecarPath.trim().isEmpty &&
        gpxPath.trim().isEmpty) {
      return const [];
    }
    return [
      SessionSegment(
        index: 1,
        videoPath: videoPath,
        sidecarPath: sidecarPath,
        gpxPath: gpxPath,
        startedAtUtc: startedAtUtc,
        endedAtUtc: endedAtUtc,
        durationMs: durationMs,
        frameCount: frameCount,
        gpsSampleCount: gpsSampleCount,
        imuSampleCount: imuSampleCount,
        totalBytes: totalBytes,
        startLat: startLat,
        startLon: startLon,
        endLat: endLat,
        endLon: endLon,
      ),
    ];
  }

  List<String> get artifactPaths {
    return [
      manifestPath,
      for (final segment in effectiveSegments) ...[
        segment.videoPath,
        segment.sidecarPath,
        segment.gpxPath,
      ],
    ].where((path) => path.trim().isNotEmpty).toList(growable: false);
  }

  SessionSummary copyWith({
    String? directoryPath,
    String? videoPath,
    String? sidecarPath,
    String? gpxPath,
    String? manifestPath,
    List<SessionSegment>? segments,
    int? totalBytes,
    String? storageLocation,
    bool? storageAvailable,
  }) {
    return SessionSummary(
      id: id,
      directoryPath: directoryPath ?? this.directoryPath,
      videoPath: videoPath ?? this.videoPath,
      sidecarPath: sidecarPath ?? this.sidecarPath,
      gpxPath: gpxPath ?? this.gpxPath,
      manifestPath: manifestPath ?? this.manifestPath,
      startedAtUtc: startedAtUtc,
      endedAtUtc: endedAtUtc,
      durationMs: durationMs,
      frameCount: frameCount,
      gpsSampleCount: gpsSampleCount,
      imuSampleCount: imuSampleCount,
      totalBytes: totalBytes ?? this.totalBytes,
      mode: mode,
      segments: segments ?? this.segments,
      storageLocation: storageLocation ?? this.storageLocation,
      storageAvailable: storageAvailable ?? this.storageAvailable,
      startLat: startLat,
      startLon: startLon,
      endLat: endLat,
      endLon: endLon,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'directory_path': directoryPath,
      'video_path': videoPath,
      'sidecar_path': sidecarPath,
      'gpx_path': gpxPath,
      'manifest_path': manifestPath,
      'started_at_utc': startedAtUtc.toIso8601String(),
      'ended_at_utc': endedAtUtc.toIso8601String(),
      'duration_ms': durationMs,
      'frame_count': frameCount,
      'gps_sample_count': gpsSampleCount,
      'imu_sample_count': imuSampleCount,
      'total_bytes': totalBytes,
      'mode': mode,
      'segments': segments.map((segment) => segment.toJson()).toList(),
      'storage_location': storageLocation,
      'start_lat': startLat,
      'start_lon': startLon,
      'end_lat': endLat,
      'end_lon': endLon,
    };
  }

  factory SessionSummary.fromJson(Map<String, dynamic> json) {
    final segments = _readSegments(json['segments']);
    return SessionSummary(
      id: json['id']?.toString() ?? '',
      directoryPath: json['directory_path']?.toString() ?? '',
      videoPath: json['video_path']?.toString() ?? '',
      sidecarPath: json['sidecar_path']?.toString() ?? '',
      gpxPath: json['gpx_path']?.toString() ?? '',
      manifestPath: json['manifest_path']?.toString() ?? '',
      startedAtUtc: _readDate(json['started_at_utc']),
      endedAtUtc: _readDate(json['ended_at_utc']),
      durationMs: _readInt(json['duration_ms']),
      frameCount: _readInt(json['frame_count']),
      gpsSampleCount: _readInt(json['gps_sample_count']),
      imuSampleCount: _readInt(json['imu_sample_count']),
      totalBytes: _readInt(json['total_bytes']),
      mode: json['mode']?.toString() ?? 'continuous',
      segments: segments,
      storageLocation: _readStorageLocation(json['storage_location']),
      startLat: _readDouble(json['start_lat']),
      startLon: _readDouble(json['start_lon']),
      endLat: _readDouble(json['end_lat']),
      endLon: _readDouble(json['end_lon']),
    );
  }
}

List<SessionSegment> _readSegments(Object? value) {
  if (value is! List) {
    return const [];
  }
  final segments = value
      .whereType<Map<String, dynamic>>()
      .map(SessionSegment.fromJson)
      .where((segment) => segment.index > 0)
      .toList();
  segments.sort((a, b) => a.index.compareTo(b.index));
  return segments;
}

Map<String, double>? _gpsJson(double? lat, double? lon) {
  if (lat == null || lon == null) {
    return null;
  }
  return {'lat': lat, 'lon': lon};
}

DateTime _readDate(Object? value) {
  if (value is String) {
    return DateTime.tryParse(value)?.toUtc() ??
        DateTime.fromMillisecondsSinceEpoch(0, isUtc: true);
  }
  return DateTime.fromMillisecondsSinceEpoch(0, isUtc: true);
}

int _readInt(Object? value) {
  if (value is num) {
    return value.round();
  }
  return 0;
}

double? _readDouble(Object? value) {
  if (value is num) {
    return value.toDouble();
  }
  return null;
}

String _readStorageLocation(Object? value) {
  return value?.toString() == 'external' ? 'external' : 'internal';
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final slash = normalized.lastIndexOf('/');
  return slash >= 0 ? normalized.substring(slash + 1) : normalized;
}
