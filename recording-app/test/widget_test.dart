import 'package:flutter_test/flutter_test.dart';
import 'package:roadsurvey_recorder/settings/app_settings.dart';

void main() {
  test('default settings match M1 scaffold requirements', () {
    final settings = AppSettings.defaults();

    expect(settings.frameSpacingM, 3);
    expect(settings.maxFps, 8);
    expect(settings.minFps, 1);
    expect(settings.pauseSpeedKmh, 2);
    expect(settings.pauseDebounceS, 3);
    expect(settings.captureMode, CaptureMode.adaptive);
    expect(settings.maxSegmentGb, 10);
    expect(settings.storageLocation, StorageLocation.auto);
    expect(settings.keepScreenOn, isTrue);
  });

  test('settings round trip through json', () {
    final settings = AppSettings.defaults().copyWith(
      captureMode: CaptureMode.continuous,
      resolution: CaptureResolution.p2160,
      codec: CaptureCodec.hevc,
      units: UnitSystem.imperial,
    );

    final roundTrip = AppSettings.fromJson(settings.toJson());

    expect(roundTrip.captureMode, CaptureMode.continuous);
    expect(roundTrip.resolution, CaptureResolution.p2160);
    expect(roundTrip.codec, CaptureCodec.hevc);
    expect(roundTrip.units, UnitSystem.imperial);
  });
}
