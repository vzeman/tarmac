import 'package:flutter_test/flutter_test.dart';
import 'package:roadsurvey_recorder/settings/app_settings.dart';

void main() {
  test('default settings match M1 scaffold requirements', () {
    final settings = AppSettings.defaults();

    expect(settings.frameSpacingM, 1);
    expect(settings.maxFps, 30);
    expect(settings.minFps, 1);
    expect(settings.autoPauseEnabled, isTrue);
    expect(settings.pauseSpeedKmh, 2);
    expect(settings.pauseDebounceS, 3);
    expect(settings.resumeSensitivity, 6);
    expect(settings.captureMode, CaptureMode.continuous);
    expect(settings.codec, CaptureCodec.hevc);
    expect(settings.maxSegmentGb, 10);
    expect(settings.storageLocation, StorageLocation.internal);
    expect(settings.keepScreenOn, isTrue);
    expect(settings.displayTheme, DisplayTheme.sunlight);
    expect(settings.autoDimWhileRecording, isFalse);
    expect(settings.mountCalibrationSet, isFalse);
    expect(settings.mountHeightM, 1.4);
    expect(settings.mountTiltDeg, 0);
    expect(settings.lensProfile, LensProfile.wide);
  });

  test('settings round trip through json', () {
    final settings = AppSettings.defaults().copyWith(
      captureMode: CaptureMode.continuous,
      resolution: CaptureResolution.p2160,
      codec: CaptureCodec.hevc,
      autoPauseEnabled: false,
      pauseSpeedKmh: 4,
      pauseDebounceS: 5,
      resumeSensitivity: 8,
      units: UnitSystem.imperial,
      displayTheme: DisplayTheme.night,
      autoDimWhileRecording: true,
      mountCalibrationSet: true,
      mountHeightM: 1.7,
      mountTiltDeg: -8,
      lensProfile: LensProfile.ultraWide,
    );

    final roundTrip = AppSettings.fromJson(settings.toJson());

    expect(roundTrip.captureMode, CaptureMode.continuous);
    expect(roundTrip.resolution, CaptureResolution.p2160);
    expect(roundTrip.codec, CaptureCodec.hevc);
    expect(roundTrip.autoPauseEnabled, isFalse);
    expect(roundTrip.pauseSpeedKmh, 4);
    expect(roundTrip.pauseDebounceS, 5);
    expect(roundTrip.resumeSensitivity, 8);
    expect(roundTrip.units, UnitSystem.imperial);
    expect(roundTrip.displayTheme, DisplayTheme.night);
    expect(roundTrip.autoDimWhileRecording, isTrue);
    expect(roundTrip.mountCalibrationSet, isTrue);
    expect(roundTrip.mountHeightM, 1.7);
    expect(roundTrip.mountTiltDeg, -8);
    expect(roundTrip.lensProfile, LensProfile.ultraWide);
  });
}
