enum CaptureMode { adaptive, continuous }

enum CaptureResolution { p720, p1080, p2160, max }

enum CaptureCodec { h264, hevc }

enum StorageLocation { auto, internal, external }

enum UnitSystem { metric, imperial }

class AppSettings {
  const AppSettings({
    required this.frameSpacingM,
    required this.maxFps,
    required this.minFps,
    required this.pauseSpeedKmh,
    required this.pauseDebounceS,
    required this.captureMode,
    required this.resolution,
    required this.codec,
    required this.maxSegmentGb,
    required this.storageLocation,
    required this.keepScreenOn,
    required this.units,
  });

  factory AppSettings.defaults() {
    return const AppSettings(
      frameSpacingM: 3,
      maxFps: 8,
      minFps: 1,
      pauseSpeedKmh: 2,
      pauseDebounceS: 3,
      captureMode: CaptureMode.adaptive,
      resolution: CaptureResolution.p1080,
      codec: CaptureCodec.h264,
      maxSegmentGb: 10,
      storageLocation: StorageLocation.auto,
      keepScreenOn: true,
      units: UnitSystem.metric,
    );
  }

  final double frameSpacingM;
  final int maxFps;
  final int minFps;
  final double pauseSpeedKmh;
  final int pauseDebounceS;
  final CaptureMode captureMode;
  final CaptureResolution resolution;
  final CaptureCodec codec;
  final double maxSegmentGb;
  final StorageLocation storageLocation;
  final bool keepScreenOn;
  final UnitSystem units;

  int get effectiveContinuousFps {
    return maxFps.clamp(minFps, 60).toInt();
  }

  AppSettings copyWith({
    double? frameSpacingM,
    int? maxFps,
    int? minFps,
    double? pauseSpeedKmh,
    int? pauseDebounceS,
    CaptureMode? captureMode,
    CaptureResolution? resolution,
    CaptureCodec? codec,
    double? maxSegmentGb,
    StorageLocation? storageLocation,
    bool? keepScreenOn,
    UnitSystem? units,
  }) {
    return AppSettings(
      frameSpacingM: frameSpacingM ?? this.frameSpacingM,
      maxFps: maxFps ?? this.maxFps,
      minFps: minFps ?? this.minFps,
      pauseSpeedKmh: pauseSpeedKmh ?? this.pauseSpeedKmh,
      pauseDebounceS: pauseDebounceS ?? this.pauseDebounceS,
      captureMode: captureMode ?? this.captureMode,
      resolution: resolution ?? this.resolution,
      codec: codec ?? this.codec,
      maxSegmentGb: maxSegmentGb ?? this.maxSegmentGb,
      storageLocation: storageLocation ?? this.storageLocation,
      keepScreenOn: keepScreenOn ?? this.keepScreenOn,
      units: units ?? this.units,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'frame_spacing_m': frameSpacingM,
      'max_fps': maxFps,
      'min_fps': minFps,
      'pause_speed_kmh': pauseSpeedKmh,
      'pause_debounce_s': pauseDebounceS,
      'capture_mode': captureMode.name,
      'resolution': resolution.name,
      'codec': codec.name,
      'max_segment_gb': maxSegmentGb,
      'storage_location': storageLocation.name,
      'keep_screen_on': keepScreenOn,
      'units': units.name,
    };
  }

  factory AppSettings.fromJson(Map<String, dynamic> json) {
    final defaults = AppSettings.defaults();
    return AppSettings(
      frameSpacingM: _readDouble(
        json['frame_spacing_m'],
        defaults.frameSpacingM,
      ),
      maxFps: _readInt(json['max_fps'], defaults.maxFps),
      minFps: _readInt(json['min_fps'], defaults.minFps),
      pauseSpeedKmh: _readDouble(
        json['pause_speed_kmh'],
        defaults.pauseSpeedKmh,
      ),
      pauseDebounceS: _readInt(
        json['pause_debounce_s'],
        defaults.pauseDebounceS,
      ),
      captureMode: _enumByName(
        CaptureMode.values,
        json['capture_mode'],
        defaults.captureMode,
      ),
      resolution: _enumByName(
        CaptureResolution.values,
        json['resolution'],
        defaults.resolution,
      ),
      codec: _enumByName(CaptureCodec.values, json['codec'], defaults.codec),
      maxSegmentGb: _readDouble(json['max_segment_gb'], defaults.maxSegmentGb),
      storageLocation: _enumByName(
        StorageLocation.values,
        json['storage_location'],
        defaults.storageLocation,
      ),
      keepScreenOn: json['keep_screen_on'] is bool
          ? json['keep_screen_on'] as bool
          : true,
      units: _enumByName(UnitSystem.values, json['units'], defaults.units),
    );
  }

  static double _readDouble(Object? value, double fallback) {
    if (value is num) {
      return value.toDouble();
    }
    return fallback;
  }

  static int _readInt(Object? value, int fallback) {
    if (value is num) {
      return value.round();
    }
    return fallback;
  }

  static T _enumByName<T extends Enum>(
    List<T> values,
    Object? raw,
    T fallback,
  ) {
    final name = raw?.toString();
    return values.cast<T?>().firstWhere(
          (value) => value?.name == name,
          orElse: () => fallback,
        ) ??
        fallback;
  }
}

extension CaptureModeLabel on CaptureMode {
  String get label {
    switch (this) {
      case CaptureMode.adaptive:
        return 'Adaptive';
      case CaptureMode.continuous:
        return 'Continuous';
    }
  }
}

extension CaptureResolutionLabel on CaptureResolution {
  String get label {
    switch (this) {
      case CaptureResolution.p720:
        return '720p';
      case CaptureResolution.p1080:
        return '1080p';
      case CaptureResolution.p2160:
        return '4K';
      case CaptureResolution.max:
        return 'Max';
    }
  }
}

extension CaptureCodecLabel on CaptureCodec {
  String get label {
    switch (this) {
      case CaptureCodec.h264:
        return 'H.264';
      case CaptureCodec.hevc:
        return 'HEVC';
    }
  }
}

extension StorageLocationLabel on StorageLocation {
  String get label {
    switch (this) {
      case StorageLocation.auto:
        return 'Auto';
      case StorageLocation.internal:
        return 'Internal';
      case StorageLocation.external:
        return 'External';
    }
  }
}

extension UnitSystemLabel on UnitSystem {
  String get label {
    switch (this) {
      case UnitSystem.metric:
        return 'Metric';
      case UnitSystem.imperial:
        return 'Imperial';
    }
  }
}
