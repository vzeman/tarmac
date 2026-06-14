import 'dart:io';

import 'package:camera/camera.dart';

import '../settings/app_settings.dart';

class CameraService {
  CameraController? _controller;
  CaptureResolution? _configuredResolution;
  int? _configuredFps;

  CameraController? get controller => _controller;

  Future<void> initialize(AppSettings settings) async {
    final existing = _controller;
    if (existing != null &&
        existing.value.isInitialized &&
        _configuredResolution == settings.resolution &&
        _configuredFps == settings.effectiveContinuousFps) {
      return;
    }

    await dispose();
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw CameraException('no_camera', 'No camera was found on this device.');
    }

    final selected = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      selected,
      _resolutionPreset(settings.resolution),
      enableAudio: false,
      fps: settings.effectiveContinuousFps,
    );
    await controller.initialize();
    _controller = controller;
    _configuredResolution = settings.resolution;
    _configuredFps = settings.effectiveContinuousFps;
  }

  Future<void> prepareRecording() async {
    final active = _requireController();
    await active.prepareForVideoRecording();
  }

  Future<void> startRecording() async {
    final active = _requireController();
    await active.startVideoRecording(enablePersistentRecording: true);
  }

  Future<XFile> stopRecording() async {
    final active = _requireController();
    return active.stopVideoRecording();
  }

  Future<File> persistRecording({
    required XFile capturedFile,
    required Directory sessionDirectory,
    required String sessionId,
  }) async {
    final extension = _extensionFor(capturedFile.path);
    final target = File(
      '${sessionDirectory.path}/${sessionId}_seg001$extension',
    );
    await capturedFile.saveTo(target.path);

    final temp = File(capturedFile.path);
    if (capturedFile.path != target.path && await temp.exists()) {
      await temp.delete();
    }
    return target;
  }

  Future<void> dispose() async {
    final existing = _controller;
    _controller = null;
    _configuredResolution = null;
    _configuredFps = null;
    await existing?.dispose();
  }

  CameraController _requireController() {
    final active = _controller;
    if (active == null || !active.value.isInitialized) {
      throw CameraException('camera_not_ready', 'Camera is not initialized.');
    }
    return active;
  }

  ResolutionPreset _resolutionPreset(CaptureResolution resolution) {
    switch (resolution) {
      case CaptureResolution.p720:
        return ResolutionPreset.high;
      case CaptureResolution.p1080:
        return ResolutionPreset.veryHigh;
      case CaptureResolution.p2160:
        return ResolutionPreset.ultraHigh;
      case CaptureResolution.max:
        return ResolutionPreset.max;
    }
  }

  String _extensionFor(String path) {
    final dot = path.lastIndexOf('.');
    if (dot >= 0 && dot < path.length - 1) {
      return path.substring(dot);
    }
    return '.mp4';
  }
}
