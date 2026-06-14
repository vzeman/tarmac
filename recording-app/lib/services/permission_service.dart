import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

class PermissionSnapshot {
  const PermissionSnapshot({
    required this.camera,
    required this.locationWhenInUse,
    required this.locationAlways,
    required this.sensors,
  });

  final PermissionStatus camera;
  final PermissionStatus locationWhenInUse;
  final PermissionStatus locationAlways;
  final PermissionStatus sensors;

  bool get canRecord => camera.isGranted && locationWhenInUse.isGranted;

  List<String> get warnings {
    return [
      if (!locationAlways.isGranted) 'Always location is not granted.',
      if (!sensors.isGranted) 'Motion permission is not granted.',
    ];
  }
}

class PermissionService {
  Future<PermissionSnapshot> requestCapturePermissions(
    BuildContext context,
  ) async {
    await _showRationale(context);
    if (!context.mounted) {
      return const PermissionSnapshot(
        camera: PermissionStatus.denied,
        locationWhenInUse: PermissionStatus.denied,
        locationAlways: PermissionStatus.denied,
        sensors: PermissionStatus.denied,
      );
    }

    final baseStatuses = await [
      Permission.camera,
      Permission.locationWhenInUse,
      Permission.sensors,
    ].request();
    final alwaysStatus = await Permission.locationAlways.request();

    final snapshot = PermissionSnapshot(
      camera: baseStatuses[Permission.camera] ?? await Permission.camera.status,
      locationWhenInUse:
          baseStatuses[Permission.locationWhenInUse] ??
          await Permission.locationWhenInUse.status,
      locationAlways: alwaysStatus,
      sensors:
          baseStatuses[Permission.sensors] ?? await Permission.sensors.status,
    );

    if (!snapshot.canRecord && context.mounted) {
      await _showSettingsPrompt(context, snapshot);
    }
    return snapshot;
  }

  Future<void> _showRationale(BuildContext context) async {
    await showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Permissions'),
          content: const Text(
            'RoadSurvey Recorder needs camera, precise location, and motion sensors to write synchronized video, GPS, and IMU sidecars.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Continue'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _showSettingsPrompt(
    BuildContext context,
    PermissionSnapshot snapshot,
  ) async {
    final cameraBlocked = snapshot.camera.isPermanentlyDenied;
    final locationBlocked = snapshot.locationWhenInUse.isPermanentlyDenied;
    await showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Permissions needed'),
          content: Text(
            cameraBlocked || locationBlocked
                ? 'Camera or foreground location is blocked in system settings.'
                : 'Camera and foreground location must be granted before recording.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
            if (cameraBlocked || locationBlocked)
              FilledButton(
                onPressed: () {
                  openAppSettings();
                  Navigator.of(context).pop();
                },
                child: const Text('Settings'),
              ),
          ],
        );
      },
    );
  }
}
