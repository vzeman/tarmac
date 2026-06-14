import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../models/telemetry.dart';
import '../services/capture_session_controller.dart';
import '../services/permission_service.dart';
import '../services/session_repository.dart';
import '../settings/app_settings.dart';

class RecordScreen extends StatefulWidget {
  const RecordScreen({
    super.key,
    required this.settings,
    required this.sessionRepository,
    required this.onSessionSaved,
  });

  final AppSettings settings;
  final SessionRepository sessionRepository;
  final Future<void> Function() onSessionSaved;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  late final CaptureSessionController _controller;
  final PermissionService _permissionService = PermissionService();
  Timer? _storageTimer;
  int? _freeBytes;

  @override
  void initState() {
    super.initState();
    _controller = CaptureSessionController(
      settings: widget.settings,
      sessionRepository: widget.sessionRepository,
    )..addListener(_handleControllerChanged);
    unawaited(_controller.initializeCamera());
    unawaited(_refreshFreeSpace());
    _storageTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => unawaited(_refreshFreeSpace()),
    );
  }

  @override
  void didUpdateWidget(covariant RecordScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.settings != widget.settings) {
      unawaited(_controller.updateSettings(widget.settings));
    }
  }

  @override
  void dispose() {
    _storageTimer?.cancel();
    _controller
      ..removeListener(_handleControllerChanged)
      ..dispose();
    super.dispose();
  }

  void _handleControllerChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _refreshFreeSpace() async {
    final bytes = await widget.sessionRepository.storageService
        .freeBytesForRecordingsRoot();
    if (mounted) {
      setState(() => _freeBytes = bytes);
    }
  }

  Future<void> _start() async {
    final permissions = await _permissionService.requestCapturePermissions(
      context,
    );
    if (!mounted) {
      return;
    }
    if (!permissions.canRecord) {
      return;
    }
    if (permissions.warnings.isNotEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(permissions.warnings.join(' '))));
    }
    await _controller.start();
  }

  Future<void> _stop() async {
    final summary = await _controller.stop();
    if (!mounted) {
      return;
    }
    if (summary != null) {
      await widget.onSessionSaved();
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Saved ${summary.id}')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        _CameraPreviewPanel(controller: _controller.cameraController),
        const SizedBox(height: 12),
        if (_controller.warningMessage != null)
          _Banner(
            icon: Icons.info_outline,
            color: theme.colorScheme.tertiaryContainer,
            message: _controller.warningMessage!,
          ),
        if (_controller.errorMessage != null)
          _Banner(
            icon: Icons.error_outline,
            color: theme.colorScheme.errorContainer,
            message: _controller.errorMessage!,
          ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed:
                    _controller.isRecording ||
                        _controller.isStopping ||
                        _controller.initializingCamera
                    ? null
                    : _start,
                icon: const Icon(Icons.fiber_manual_record),
                label: const Text('Start'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _controller.isRecording && !_controller.isStopping
                    ? _stop
                    : null,
                icon: const Icon(Icons.stop),
                label: Text(_controller.isStopping ? 'Stopping' : 'Stop'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _MetricTile(
              icon: Icons.speed,
              label: 'Speed',
              value: _formatSpeed(_controller.speedMps, widget.settings.units),
            ),
            _MetricTile(
              icon: Icons.satellite_alt,
              label: 'GPS',
              value: _gpsFixText(_controller.gpsAccuracyM),
            ),
            _MetricTile(
              icon: Icons.timer_outlined,
              label: 'Elapsed',
              value: _formatDuration(_controller.elapsed),
            ),
            _MetricTile(
              icon: Icons.image_outlined,
              label: 'Frames',
              value: _controller.estimatedFrameCount.toString(),
            ),
            _MetricTile(
              icon: Icons.my_location,
              label: 'GPS samples',
              value: _controller.gpsSamples.toString(),
            ),
            _MetricTile(
              icon: Icons.sensors,
              label: 'IMU samples',
              value: _controller.imuSamples.toString(),
            ),
            _MetricTile(
              icon: Icons.storage,
              label: 'Free',
              value: _formatBytes(_freeBytes),
            ),
            _MetricTile(
              icon: Icons.movie_creation_outlined,
              label: 'Mode',
              value: 'Continuous',
            ),
          ],
        ),
        const SizedBox(height: 12),
        SizedBox(height: 190, child: _LiveMap(points: _controller.track)),
      ],
    );
  }
}

class _CameraPreviewPanel extends StatelessWidget {
  const _CameraPreviewPanel({required this.controller});

  final CameraController? controller;

  @override
  Widget build(BuildContext context) {
    final active = controller;
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: Container(
          color: Colors.black,
          alignment: Alignment.center,
          child: active != null && active.value.isInitialized
              ? CameraPreview(active)
              : const Icon(Icons.videocam_off, color: Colors.white54, size: 48),
        ),
      ),
    );
  }
}

class _LiveMap extends StatelessWidget {
  const _LiveMap({required this.points});

  final List<TrackPoint> points;

  @override
  Widget build(BuildContext context) {
    final latLngs = points
        .map((point) => LatLng(point.lat, point.lon))
        .toList();
    final center = latLngs.isEmpty ? const LatLng(0, 0) : latLngs.last;
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: FlutterMap(
        key: ValueKey(
          '${latLngs.length}_${center.latitude}_${center.longitude}',
        ),
        options: MapOptions(
          initialCenter: center,
          initialZoom: latLngs.isEmpty ? 2 : 16,
        ),
        children: [
          TileLayer(
            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            userAgentPackageName: 'com.qualityunit.roadsurvey_recorder',
          ),
          if (latLngs.length > 1)
            PolylineLayer(
              polylines: [
                Polyline(
                  points: latLngs,
                  strokeWidth: 4,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ],
            ),
          if (latLngs.isNotEmpty)
            MarkerLayer(
              markers: [
                Marker(
                  point: latLngs.last,
                  width: 34,
                  height: 34,
                  child: const Icon(Icons.navigation, color: Colors.redAccent),
                ),
              ],
            ),
        ],
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SizedBox(
      width: 164,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              Icon(icon, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label, style: theme.textTheme.labelSmall),
                    Text(
                      value,
                      style: theme.textTheme.titleSmall,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({
    required this.icon,
    required this.color,
    required this.message,
  });

  final IconData icon;
  final Color color;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              Icon(icon),
              const SizedBox(width: 8),
              Expanded(child: Text(message)),
            ],
          ),
        ),
      ),
    );
  }
}

String _formatSpeed(double speedMps, UnitSystem units) {
  if (units == UnitSystem.imperial) {
    return '${(speedMps * 2.236936).toStringAsFixed(1)} mph';
  }
  return '${(speedMps * 3.6).toStringAsFixed(1)} km/h';
}

String _gpsFixText(double? accuracy) {
  if (accuracy == null) {
    return 'No fix';
  }
  return '${accuracy.toStringAsFixed(1)} m';
}

String _formatDuration(Duration duration) {
  final hours = duration.inHours;
  final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
  final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
  return '$hours:$minutes:$seconds';
}

String _formatBytes(int? bytes) {
  if (bytes == null) {
    return 'Unknown';
  }
  const gb = 1024 * 1024 * 1024;
  const mb = 1024 * 1024;
  if (bytes >= gb) {
    return '${(bytes / gb).toStringAsFixed(1)} GB';
  }
  return '${(bytes / mb).toStringAsFixed(0)} MB';
}
