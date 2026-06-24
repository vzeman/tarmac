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

class LidarFrame {
  const LidarFrame({
    required this.utcMs,
    required this.ptsMs,
    required this.pose,
    required this.fx,
    required this.fy,
    required this.cx,
    required this.cy,
    required this.imgW,
    required this.imgH,
    required this.roughness,
    required this.vertAccelMps2,
    required this.tracking,
    this.depthF32,
    this.depthW,
    this.depthH,
  });

  // Wall-clock timestamp and presentation timestamp (relative to segment start).
  final int utcMs;
  final int ptsMs;

  // ARKit camera-to-world 4×4 transform, column-major (16 doubles).
  final List<double> pose;

  // Camera intrinsics from ARKit.
  final double fx;
  final double fy;
  final double cx;
  final double cy;
  final int imgW;
  final int imgH;

  // Std-dev of centre depth patch (metres). 0 when no LiDAR.
  final double roughness;

  // Gravity-corrected vertical acceleration in m/s² (world Y-up).
  // Positive = upward jolt (landing after pothole), negative = dropping in.
  final double vertAccelMps2;

  // "normal" | "limited:*" | "notAvailable"
  final String tracking;

  // Optional LiDAR depth: base64 Float32 array, row-major.
  final String? depthF32;
  final int? depthW;
  final int? depthH;

  factory LidarFrame.fromNative(Map<String, dynamic> m) {
    final rawPose = m['pose'] as List<dynamic>;
    return LidarFrame(
      utcMs: (m['utc_ms'] as num).toInt(),
      ptsMs: (m['pts_ms'] as num).toInt(),
      pose: rawPose.map((v) => (v as num).toDouble()).toList(),
      fx: (m['fx'] as num).toDouble(),
      fy: (m['fy'] as num).toDouble(),
      cx: (m['cx'] as num).toDouble(),
      cy: (m['cy'] as num).toDouble(),
      imgW: (m['img_w'] as num).toInt(),
      imgH: (m['img_h'] as num).toInt(),
      roughness: (m['roughness'] as num?)?.toDouble() ?? 0.0,
      vertAccelMps2: (m['vert_accel'] as num?)?.toDouble() ?? 0.0,
      tracking: (m['tracking'] as String?) ?? 'unknown',
      depthF32: m['depth_f32'] as String?,
      depthW: (m['depth_w'] as num?)?.toInt(),
      depthH: (m['depth_h'] as num?)?.toInt(),
    );
  }

  Map<String, dynamic> toJson() => {
    'utc_ms': utcMs,
    'pts_ms': ptsMs,
    'pose': pose,
    'fx': fx,
    'fy': fy,
    'cx': cx,
    'cy': cy,
    'img_w': imgW,
    'img_h': imgH,
    'roughness': roughness,
    'vert_accel': vertAccelMps2,
    'tracking': tracking,
    if (depthF32 != null) 'depth_f32': depthF32,
    if (depthW != null) 'depth_w': depthW,
    if (depthH != null) 'depth_h': depthH,
  };
}

class LidarPoint {
  const LidarPoint({
    required this.ptsMs,
    required this.roughness,
    required this.vertAccelMps2,
  });

  final int ptsMs;
  final double roughness;
  final double vertAccelMps2;
}
