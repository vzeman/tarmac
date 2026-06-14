enum CaptureMode { adaptive, continuous }

enum CaptureResolution { p720, p1080, p2160, max }

enum CaptureCodec { h264, hevc }

enum StorageLocation { auto, internal, external }

enum UnitSystem { metric, imperial }

enum DisplayTheme { sunlight, night }

enum LensProfile { wide, ultraWide, telephoto }

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
    required this.displayTheme,
    required this.autoDimWhileRecording,
    required this.mountCalibrationSet,
    required this.mountHeightM,
    required this.mountTiltDeg,
    required this.lensProfile,
  });

  factory AppSettings.defaults() {
    return const AppSettings(
      frameSpacingM: 1,
      maxFps: 30,
      minFps: 1,
      pauseSpeedKmh: 2,
      pauseDebounceS: 3,
      captureMode: CaptureMode.continuous,
      resolution: CaptureResolution.p1080,
      codec: CaptureCodec.h264,
      maxSegmentGb: 10,
      storageLocation: StorageLocation.auto,
      keepScreenOn: true,
      units: UnitSystem.metric,
      displayTheme: DisplayTheme.sunlight,
      autoDimWhileRecording: false,
      mountCalibrationSet: false,
      mountHeightM: 1.4,
      mountTiltDeg: 0,
      lensProfile: LensProfile.wide,
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
  final DisplayTheme displayTheme;
  final bool autoDimWhileRecording;
  final bool mountCalibrationSet;
  final double mountHeightM;
  final double mountTiltDeg;
  final LensProfile lensProfile;

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
    DisplayTheme? displayTheme,
    bool? autoDimWhileRecording,
    bool? mountCalibrationSet,
    double? mountHeightM,
    double? mountTiltDeg,
    LensProfile? lensProfile,
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
      displayTheme: displayTheme ?? this.displayTheme,
      autoDimWhileRecording:
          autoDimWhileRecording ?? this.autoDimWhileRecording,
      mountCalibrationSet: mountCalibrationSet ?? this.mountCalibrationSet,
      mountHeightM: mountHeightM ?? this.mountHeightM,
      mountTiltDeg: mountTiltDeg ?? this.mountTiltDeg,
      lensProfile: lensProfile ?? this.lensProfile,
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
      'display_theme': displayTheme.name,
      'auto_dim_while_recording': autoDimWhileRecording,
      'mount_calibration_set': mountCalibrationSet,
      'mount_height_m': mountHeightM,
      'mount_tilt_deg': mountTiltDeg,
      'lens_profile': lensProfile.name,
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
      displayTheme: _enumByName(
        DisplayTheme.values,
        json['display_theme'],
        defaults.displayTheme,
      ),
      autoDimWhileRecording: json['auto_dim_while_recording'] is bool
          ? json['auto_dim_while_recording'] as bool
          : defaults.autoDimWhileRecording,
      mountCalibrationSet: json['mount_calibration_set'] is bool
          ? json['mount_calibration_set'] as bool
          : defaults.mountCalibrationSet,
      mountHeightM: _readDouble(json['mount_height_m'], defaults.mountHeightM),
      mountTiltDeg: _readDouble(json['mount_tilt_deg'], defaults.mountTiltDeg),
      lensProfile: _enumByName(
        LensProfile.values,
        json['lens_profile'],
        defaults.lensProfile,
      ),
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

extension DisplayThemeLabel on DisplayTheme {
  String get label {
    switch (this) {
      case DisplayTheme.sunlight:
        return 'Sunlight';
      case DisplayTheme.night:
        return 'Night';
    }
  }
}

extension LensProfileLabel on LensProfile {
  String get label {
    switch (this) {
      case LensProfile.wide:
        return 'Wide';
      case LensProfile.ultraWide:
        return 'Ultra wide';
      case LensProfile.telephoto:
        return 'Telephoto';
    }
  }
}
