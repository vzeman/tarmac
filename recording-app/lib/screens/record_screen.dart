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
    return OrientationBuilder(
      builder: (context, orientation) {
        return LayoutBuilder(
          builder: (context, constraints) {
            final isLandscape =
                orientation == Orientation.landscape ||
                constraints.maxWidth > constraints.maxHeight;
            if (isLandscape) {
              return _buildLandscapeRecord(context, constraints);
            }
            return _buildPortraitRecord(context);
          },
        );
      },
    );
  }

  Widget _buildPortraitRecord(BuildContext context) {
    final theme = Theme.of(context);
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(12, 12, 12, 20),
        children: [
          _CameraPreviewPanel(controller: _controller.cameraController),
          const SizedBox(height: 12),
          ..._buildBanners(theme),
          const SizedBox(height: 8),
          _buildActionButtons(),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _metrics()
                .map(
                  (metric) => _MetricTile(
                    icon: metric.icon,
                    label: metric.label,
                    value: metric.value,
                    width: 164,
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 12),
          SizedBox(height: 190, child: _LiveMap(points: _controller.track)),
        ],
      ),
    );
  }

  Widget _buildLandscapeRecord(
    BuildContext context,
    BoxConstraints constraints,
  ) {
    final theme = Theme.of(context);
    final railWidth = (constraints.maxWidth * 0.34).clamp(286.0, 360.0);
    final mapWidth = (constraints.maxWidth * 0.22).clamp(150.0, 220.0);
    final mapHeight = (constraints.maxHeight * 0.3).clamp(86.0, 128.0);
    final banners = _buildBanners(theme, compact: true);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Row(
          children: [
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  _CameraPreviewPanel(
                    controller: _controller.cameraController,
                    fill: true,
                  ),
                  if (banners.isNotEmpty)
                    Positioned(
                      left: 10,
                      top: 10,
                      right: 10,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: banners,
                      ),
                    ),
                  Positioned(
                    left: 10,
                    bottom: 10,
                    child: SizedBox(
                      width: mapWidth,
                      height: mapHeight,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface.withAlpha(225),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                            color: theme.colorScheme.outlineVariant,
                          ),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(2),
                          child: _LiveMap(points: _controller.track),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(width: railWidth, child: _buildLandscapeRail(context)),
          ],
        ),
      ),
    );
  }

  Widget _buildLandscapeRail(BuildContext context) {
    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withAlpha(236),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final compactHeight = constraints.maxHeight < 340;
            return Column(
              children: [
                _buildActionButtons(compact: true),
                const SizedBox(height: 8),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, metricConstraints) {
                      return FittedBox(
                        fit: BoxFit.scaleDown,
                        alignment: Alignment.topCenter,
                        child: SizedBox(
                          width: metricConstraints.maxWidth,
                          child: _MetricGrid(
                            metrics: _metrics(),
                            tileHeight: compactHeight ? 40 : 48,
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildActionButtons({bool compact = false}) {
    final buttonHeight = compact ? 42.0 : 48.0;
    final canStart =
        !_controller.isRecording &&
        !_controller.isStopping &&
        !_controller.initializingCamera;
    final canStop = _controller.isRecording && !_controller.isStopping;

    return SizedBox(
      height: buttonHeight,
      child: Row(
        children: [
          Expanded(
            child: FilledButton.icon(
              onPressed: canStart ? _start : null,
              icon: const Icon(Icons.fiber_manual_record),
              label: const Text('Start'),
              style: FilledButton.styleFrom(
                minimumSize: Size(0, buttonHeight),
                padding: const EdgeInsets.symmetric(horizontal: 10),
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: canStop ? _stop : null,
              icon: const Icon(Icons.stop),
              label: Text(_controller.isStopping ? 'Stopping' : 'Stop'),
              style: OutlinedButton.styleFrom(
                minimumSize: Size(0, buttonHeight),
                padding: const EdgeInsets.symmetric(horizontal: 10),
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildBanners(ThemeData theme, {bool compact = false}) {
    return [
      if (_controller.warningMessage != null)
        _Banner(
          icon: Icons.info_outline,
          color: theme.colorScheme.tertiaryContainer,
          message: _controller.warningMessage!,
          compact: compact,
        ),
      if (_controller.errorMessage != null)
        _Banner(
          icon: Icons.error_outline,
          color: theme.colorScheme.errorContainer,
          message: _controller.errorMessage!,
          compact: compact,
        ),
    ];
  }

  List<_MetricData> _metrics() {
    return [
      _MetricData(
        icon: Icons.speed,
        label: 'Speed',
        value: _formatSpeed(_controller.speedMps, widget.settings.units),
      ),
      _MetricData(
        icon: Icons.satellite_alt,
        label: 'GPS fix',
        value: _gpsFixText(_controller.gpsAccuracyM),
      ),
      _MetricData(
        icon: Icons.timer_outlined,
        label: 'Elapsed',
        value: _formatDuration(_controller.elapsed),
      ),
      _MetricData(
        icon: Icons.image_outlined,
        label: 'Frames',
        value: _controller.estimatedFrameCount.toString(),
      ),
      _MetricData(
        icon: Icons.my_location,
        label: 'GPS samples',
        value: _controller.gpsSamples.toString(),
      ),
      _MetricData(
        icon: Icons.sensors,
        label: 'IMU samples',
        value: _controller.imuSamples.toString(),
      ),
      _MetricData(
        icon: Icons.storage,
        label: 'Free',
        value: _formatBytes(_freeBytes),
      ),
      const _MetricData(
        icon: Icons.movie_creation_outlined,
        label: 'Mode',
        value: 'Continuous',
      ),
    ];
  }
}

class _MetricData {
  const _MetricData({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;
}

class _MetricGrid extends StatelessWidget {
  const _MetricGrid({required this.metrics, required this.tileHeight});

  final List<_MetricData> metrics;
  final double tileHeight;

  @override
  Widget build(BuildContext context) {
    const spacing = 6.0;
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth >= 250 ? 2 : 1;
        final tileWidth =
            (constraints.maxWidth - (spacing * (columns - 1))) / columns;
        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: metrics
              .map(
                (metric) => _MetricTile(
                  icon: metric.icon,
                  label: metric.label,
                  value: metric.value,
                  width: tileWidth,
                  height: tileHeight,
                  compact: true,
                ),
              )
              .toList(),
        );
      },
    );
  }
}

class _CameraPreviewPanel extends StatelessWidget {
  const _CameraPreviewPanel({required this.controller, this.fill = false});

  final CameraController? controller;
  final bool fill;

  @override
  Widget build(BuildContext context) {
    final active = controller;
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Container(
        color: Colors.black,
        alignment: Alignment.center,
        child: active != null && active.value.isInitialized
            ? _CameraPreviewContent(controller: active, fill: fill)
            : const Icon(Icons.videocam_off, color: Colors.white54, size: 48),
      ),
    );
  }
}

class _CameraPreviewContent extends StatelessWidget {
  const _CameraPreviewContent({required this.controller, required this.fill});

  final CameraController controller;
  final bool fill;

  @override
  Widget build(BuildContext context) {
    if (!fill) {
      return AspectRatio(aspectRatio: 16 / 9, child: CameraPreview(controller));
    }

    final aspectRatio = controller.value.aspectRatio;
    final previewWidth = aspectRatio >= 1 ? aspectRatio : 1.0;
    final previewHeight = aspectRatio >= 1 ? 1.0 : 1 / aspectRatio;
    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: previewWidth,
          height: previewHeight,
          child: CameraPreview(controller),
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
    this.width,
    this.height,
    this.compact = false,
  });

  final IconData icon;
  final String label;
  final String value;
  final double? width;
  final double? height;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SizedBox(
      width: width,
      height: height,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: compact ? 8 : 10,
            vertical: compact ? 6 : 10,
          ),
          child: Row(
            children: [
              Icon(icon, size: compact ? 18 : 20),
              SizedBox(width: compact ? 6 : 8),
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: theme.textTheme.labelSmall,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      value,
                      style: compact
                          ? theme.textTheme.bodySmall
                          : theme.textTheme.titleSmall,
                      maxLines: 1,
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
    this.compact = false,
  });

  final IconData icon;
  final Color color;
  final String message;
  final bool compact;

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
          padding: EdgeInsets.all(compact ? 8 : 10),
          child: Row(
            children: [
              Icon(icon, size: compact ? 20 : 24),
              SizedBox(width: compact ? 6 : 8),
              Expanded(
                child: Text(
                  message,
                  maxLines: compact ? 2 : null,
                  overflow: compact ? TextOverflow.ellipsis : null,
                ),
              ),
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
