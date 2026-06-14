class TrackPoint {
  const TrackPoint({required this.lat, required this.lon, this.utcMs});

  final double lat;
  final double lon;
  final int? utcMs;
}

class GpsSample {
  const GpsSample({
    required this.utcMs,
    required this.ptsMs,
    required this.fixUtcMs,
    required this.lat,
    required this.lon,
    required this.altM,
    required this.horizontalAccuracyM,
    required this.verticalAccuracyM,
    required this.speedMps,
    required this.headingDeg,
  });

  final int utcMs;
  final int ptsMs;
  final int fixUtcMs;
  final double lat;
  final double lon;
  final double? altM;
  final double? horizontalAccuracyM;
  final double? verticalAccuracyM;
  final double? speedMps;
  final double? headingDeg;

  Map<String, dynamic> toJson() {
    return {
      'utc_ms': utcMs,
      'pts_ms': ptsMs,
      'fix_utc_ms': fixUtcMs,
      'lat': lat,
      'lon': lon,
      'alt_m': altM,
      'horizontal_accuracy_m': horizontalAccuracyM,
      'vertical_accuracy_m': verticalAccuracyM,
      'speed_mps': speedMps,
      'heading_deg': headingDeg,
    };
  }

  factory GpsSample.fromJson(Map<String, dynamic> json) {
    return GpsSample(
      utcMs: _readInt(json['utc_ms']),
      ptsMs: _readInt(json['pts_ms']),
      fixUtcMs: _readInt(json['fix_utc_ms']),
      lat: _readDouble(json['lat']) ?? 0,
      lon: _readDouble(json['lon']) ?? 0,
      altM: _readDouble(json['alt_m']),
      horizontalAccuracyM: _readDouble(json['horizontal_accuracy_m']),
      verticalAccuracyM: _readDouble(json['vertical_accuracy_m']),
      speedMps: _readDouble(json['speed_mps']),
      headingDeg: _readDouble(json['heading_deg']),
    );
  }
}

class ImuSample {
  const ImuSample({
    required this.utcMs,
    required this.ptsMs,
    required this.ax,
    required this.ay,
    required this.az,
    required this.gx,
    required this.gy,
    required this.gz,
  });

  final int utcMs;
  final int ptsMs;
  final double ax;
  final double ay;
  final double az;
  final double gx;
  final double gy;
  final double gz;

  Map<String, dynamic> toJson() {
    return {
      'utc_ms': utcMs,
      'pts_ms': ptsMs,
      'ax': ax,
      'ay': ay,
      'az': az,
      'gx': gx,
      'gy': gy,
      'gz': gz,
    };
  }
}

class FrameSample {
  const FrameSample({
    required this.frameIndex,
    required this.ptsMs,
    required this.utcMs,
    required this.lat,
    required this.lon,
    required this.altM,
    required this.gpsAccuracyM,
    required this.speedMps,
    required this.headingDeg,
  });

  final int frameIndex;
  final int ptsMs;
  final int utcMs;
  final double? lat;
  final double? lon;
  final double? altM;
  final double? gpsAccuracyM;
  final double? speedMps;
  final double? headingDeg;

  Map<String, dynamic> toJson() {
    return {
      'frame_index': frameIndex,
      'pts_ms': ptsMs,
      'utc_ms': utcMs,
      'lat': lat,
      'lon': lon,
      'alt_m': altM,
      'gps_accuracy_m': gpsAccuracyM,
      'speed_mps': speedMps,
      'heading_deg': headingDeg,
    };
  }
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
