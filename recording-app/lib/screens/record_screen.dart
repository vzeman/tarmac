import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../models/telemetry.dart';
import '../services/capture_session_controller.dart';
import '../services/permission_service.dart';
import '../services/session_repository.dart';
import '../services/storage_service.dart';
import '../settings/app_settings.dart';

const _readyColor = Color(0xFF18B85F);
const _recordingColor = Color(0xFFE11931);
const _warningColor = Color(0xFFFFB000);
const _idleColor = Color(0xFF7A838C);

enum _SurveyVisualState { ready, recording, paused, stopping, idle }

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

class _RecordScreenState extends State<RecordScreen>
    with SingleTickerProviderStateMixin {
  static const _stopHoldDuration = Duration(milliseconds: 1250);

  late final CaptureSessionController _controller;
  late final AnimationController _pulseController;
  final PermissionService _permissionService = PermissionService();
  Timer? _storageTimer;
  Timer? _preflightGpsTimer;
  Timer? _stopHoldTimer;
  int? _freeBytes;
  StorageTarget? _storageTarget;
  double? _preflightGpsAccuracyM;
  DateTime? _preflightGpsFixUtc;
  bool _checkingPreflight = false;
  bool _mapExpanded = false;
  double _stopHoldProgress = 0;
  _SurveyVisualState? _lastVisualState;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
      lowerBound: 0.0,
      upperBound: 1.0,
    )..repeat(reverse: true);
    _controller = CaptureSessionController(
      settings: widget.settings,
      sessionRepository: widget.sessionRepository,
    )..addListener(_handleControllerChanged);
    _lastVisualState = _visualState();
    unawaited(_controller.initializeCamera());
    unawaited(_refreshFreeSpace());
    unawaited(_refreshPreflightGps());
    _storageTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => unawaited(_refreshFreeSpace()),
    );
    _preflightGpsTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => unawaited(_refreshPreflightGps()),
    );
  }

  @override
  void didUpdateWidget(covariant RecordScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.settings != widget.settings) {
      unawaited(_controller.updateSettings(widget.settings));
      unawaited(_refreshFreeSpace());
    }
  }

  @override
  void dispose() {
    _storageTimer?.cancel();
    _preflightGpsTimer?.cancel();
    _cancelStopHold(resetProgress: false);
    _pulseController.dispose();
    _controller
      ..removeListener(_handleControllerChanged)
      ..dispose();
    super.dispose();
  }

  void _handleControllerChanged() {
    final nextState = _visualState();
    if (_lastVisualState == _SurveyVisualState.paused &&
        nextState != _SurveyVisualState.paused) {
      unawaited(HapticFeedback.selectionClick());
    } else if (_lastVisualState != _SurveyVisualState.paused &&
        nextState == _SurveyVisualState.paused) {
      unawaited(HapticFeedback.mediumImpact());
    }
    _lastVisualState = nextState;
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _refreshFreeSpace() async {
    final target = await widget.sessionRepository.storageService.activeTarget(
      widget.settings,
    );
    if (mounted) {
      setState(() {
        _storageTarget = target;
        _freeBytes = target.freeBytes;
      });
    }
  }

  Future<void> _refreshPreflightGps() async {
    if (_controller.isRecording || _checkingPreflight) {
      return;
    }
    try {
      final fix = await _controller.locationService.currentBestFix();
      if (!mounted || fix == null) {
        return;
      }
      setState(() {
        _preflightGpsAccuracyM = fix.accuracy.isFinite ? fix.accuracy : null;
        _preflightGpsFixUtc = fix.timestamp.toUtc();
      });
    } on Exception {
      if (mounted) {
        setState(() {
          _preflightGpsAccuracyM = null;
          _preflightGpsFixUtc = null;
        });
      }
    }
  }

  Future<void> _runPreflightCheck() async {
    if (_checkingPreflight) {
      return;
    }
    setState(() => _checkingPreflight = true);
    final permissions = await _permissionService.requestCapturePermissions(
      context,
    );
    if (!mounted) {
      return;
    }
    if (permissions.canRecord) {
      await _controller.initializeCamera();
      await _refreshFreeSpace();
      await _refreshPreflightGps();
    }
    if (!mounted) {
      return;
    }
    if (permissions.warnings.isNotEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(permissions.warnings.join(' '))));
    }
    setState(() => _checkingPreflight = false);
  }

  Future<void> _start() async {
    final readiness = _readiness();
    if (!readiness.canStart) {
      await _runPreflightCheck();
      if (!mounted) {
        return;
      }
      if (!_readiness().canStart) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Complete pre-flight checks first.')),
        );
        return;
      }
    }

    StorageTargetType? storageTargetOverride;
    if (widget.settings.storageLocation == StorageLocation.external) {
      final available = await widget.sessionRepository.storageService
          .externalAvailable();
      if (!mounted) {
        return;
      }
      if (!available) {
        final continueInternal = await showDialog<bool>(
          context: context,
          builder: (context) {
            return AlertDialog(
              title: const Text("External storage isn't connected."),
              content: const Text('Continue recording to internal storage?'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(context).pop(true),
                  child: const Text('Continue on Internal'),
                ),
              ],
            );
          },
        );
        if (continueInternal != true) {
          return;
        }
        storageTargetOverride = StorageTargetType.internal;
        await _refreshFreeSpace();
        if (!mounted) {
          return;
        }
      }
    }

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
    if (!widget.settings.mountCalibrationSet) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Starting with mount calibration marked set later.'),
        ),
      );
    }
    await HapticFeedback.mediumImpact();
    await _controller.start(storageTargetOverride: storageTargetOverride);
    if (_controller.isRecording) {
      await HapticFeedback.heavyImpact();
    }
  }

  Future<void> _stop() async {
    if (!_controller.isRecording || _controller.isStopping) {
      return;
    }
    _cancelStopHold(resetProgress: false);
    setState(() => _stopHoldProgress = 1);
    await HapticFeedback.heavyImpact();
    final summary = await _controller.stop();
    if (!mounted) {
      return;
    }
    setState(() => _stopHoldProgress = 0);
    await _refreshFreeSpace();
    if (summary != null) {
      await widget.onSessionSaved();
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Saved ${summary.id}')));
      }
    }
  }

  void _beginStopHold() {
    if (!_controller.isRecording || _controller.isStopping) {
      return;
    }
    _cancelStopHold(resetProgress: false);
    final started = DateTime.now();
    setState(() => _stopHoldProgress = 0);
    unawaited(HapticFeedback.selectionClick());
    _stopHoldTimer = Timer.periodic(const Duration(milliseconds: 16), (timer) {
      final elapsed = DateTime.now().difference(started);
      final progress =
          elapsed.inMilliseconds / _stopHoldDuration.inMilliseconds;
      if (progress >= 1) {
        timer.cancel();
        _stopHoldTimer = null;
        unawaited(_stop());
        return;
      }
      if (mounted) {
        setState(() => _stopHoldProgress = progress.clamp(0.0, 1.0));
      }
    });
  }

  void _cancelStopHold({bool resetProgress = true}) {
    _stopHoldTimer?.cancel();
    _stopHoldTimer = null;
    if (resetProgress && mounted && _stopHoldProgress != 0) {
      setState(() => _stopHoldProgress = 0);
    }
  }

  _SurveyVisualState _visualState() {
    if (_controller.isStopping) {
      return _SurveyVisualState.stopping;
    }
    if (_controller.isRecording) {
      return _isStationaryPauseVisual
          ? _SurveyVisualState.paused
          : _SurveyVisualState.recording;
    }
    final readiness = _readiness();
    return readiness.canStart
        ? _SurveyVisualState.ready
        : _SurveyVisualState.idle;
  }

  bool get _isStationaryPauseVisual {
    final speedKmh = _controller.speedMps * 3.6;
    return widget.settings.captureMode == CaptureMode.adaptive &&
        _controller.elapsed.inSeconds >= widget.settings.pauseDebounceS &&
        speedKmh <= widget.settings.pauseSpeedKmh;
  }

  _PreflightReadiness _readiness() {
    final storageTime = _estimatedTimeLeft(_freeBytes, widget.settings);
    final accuracy = _effectiveGpsAccuracy;
    final cameraReady =
        _controller.cameraController?.value.isInitialized == true &&
        !_controller.initializingCamera &&
        _controller.errorMessage == null;
    final gpsReady = accuracy != null && accuracy <= 15;
    final storageReady =
        storageTime != null && storageTime >= const Duration(minutes: 20);
    return _PreflightReadiness(
      cameraReady: cameraReady,
      gpsReady: gpsReady,
      storageReady: storageReady,
      calibrationSet: widget.settings.mountCalibrationSet,
      storageTimeLeft: storageTime,
      storageDetail: _storageDetail(_storageTarget, storageTime),
      gpsAccuracyM: accuracy,
      gpsFixUtc: _controller.lastGpsFixUtc ?? _preflightGpsFixUtc,
    );
  }

  double? get _effectiveGpsAccuracy {
    return _controller.gpsAccuracyM ?? _preflightGpsAccuracyM;
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

  Widget _buildLandscapeRecord(
    BuildContext context,
    BoxConstraints constraints,
  ) {
    final railWidth = (constraints.maxWidth * 0.2).clamp(164.0, 214.0);
    final mapWidth = _mapExpanded
        ? (constraints.maxWidth * 0.42).clamp(300.0, 480.0)
        : (constraints.maxWidth * 0.24).clamp(210.0, 280.0);
    final mapHeight = _mapExpanded
        ? (constraints.maxHeight * 0.45).clamp(190.0, 300.0)
        : (constraints.maxHeight * 0.28).clamp(130.0, 180.0);
    final readiness = _readiness();

    return SafeArea(
      child: Stack(
        fit: StackFit.expand,
        children: [
          _CameraPreviewPanel(
            controller: _controller.cameraController,
            fill: true,
          ),
          const _HudScrim(),
          if (widget.settings.autoDimWhileRecording && _controller.isRecording)
            const _DimOverlay(),
          Positioned(
            top: 12,
            left: 16,
            right: railWidth + 28,
            child: _TopStrip(
              gpsAccuracyM: _effectiveGpsAccuracy,
              satelliteCount: null,
              elapsed: _controller.elapsed,
              storageTimeLeft: readiness.storageTimeLeft,
              gpsReady: readiness.gpsReady,
            ),
          ),
          Positioned(
            top: 60,
            left: 0,
            right: railWidth + 20,
            child: Center(
              child: _StatePill(
                state: _visualState(),
                animation: _pulseController,
              ),
            ),
          ),
          Positioned(
            left: 22,
            top: constraints.maxHeight * 0.24,
            child: _SpeedReadout(
              speedMps: _controller.speedMps,
              units: widget.settings.units,
            ),
          ),
          Positioned(
            top: 12,
            right: 14,
            bottom: 14,
            width: railWidth,
            child: _ThumbRail(
              controller: _controller,
              settings: widget.settings,
              freeBytes: _freeBytes,
              storageTarget: _storageTarget,
              stopHoldProgress: _stopHoldProgress,
              onStart: _start,
              onStopHoldStart: _beginStopHold,
              onStopHoldCancel: _cancelStopHold,
            ),
          ),
          Positioned(
            left: 18,
            bottom: 18,
            width: mapWidth,
            height: mapHeight,
            child: _MapInset(
              points: _controller.track,
              expanded: _mapExpanded,
              onToggle: () => setState(() => _mapExpanded = !_mapExpanded),
            ),
          ),
          if (!_controller.isRecording)
            Positioned(
              left: math.min(mapWidth + 34, constraints.maxWidth * 0.36),
              right: railWidth + 28,
              bottom: 18,
              child: _PreflightSheet(
                readiness: readiness,
                checking: _checkingPreflight,
                onCheck: _runPreflightCheck,
                onStart: readiness.canStart ? _start : null,
              ),
            ),
          if (_controller.warningMessage != null ||
              _controller.errorMessage != null)
            Positioned(
              left: 18,
              right: railWidth + 28,
              top: 112,
              child: _MessageStack(
                warning: _controller.warningMessage,
                error: _controller.errorMessage,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPortraitRecord(BuildContext context) {
    final readiness = _readiness();
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(12, 12, 12, 24),
        children: [
          AspectRatio(
            aspectRatio: 16 / 9,
            child: Stack(
              fit: StackFit.expand,
              children: [
                _CameraPreviewPanel(
                  controller: _controller.cameraController,
                  fill: true,
                ),
                const _HudScrim(),
                if (widget.settings.autoDimWhileRecording &&
                    _controller.isRecording)
                  const _DimOverlay(),
                Positioned(
                  left: 8,
                  right: 8,
                  top: 8,
                  child: _TopStrip(
                    gpsAccuracyM: _effectiveGpsAccuracy,
                    satelliteCount: null,
                    elapsed: _controller.elapsed,
                    storageTimeLeft: readiness.storageTimeLeft,
                    gpsReady: readiness.gpsReady,
                  ),
                ),
                Positioned(
                  left: 0,
                  right: 0,
                  top: 58,
                  child: Center(
                    child: _StatePill(
                      state: _visualState(),
                      animation: _pulseController,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          _MessageStack(
            warning: _controller.warningMessage,
            error: _controller.errorMessage,
          ),
          if (!_controller.isRecording) ...[
            _PreflightSheet(
              readiness: readiness,
              checking: _checkingPreflight,
              onCheck: _runPreflightCheck,
              onStart: readiness.canStart ? _start : null,
            ),
            const SizedBox(height: 12),
          ],
          _SpeedReadout(
            speedMps: _controller.speedMps,
            units: widget.settings.units,
            compact: true,
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 128,
            child: _ThumbRail(
              controller: _controller,
              settings: widget.settings,
              freeBytes: _freeBytes,
              storageTarget: _storageTarget,
              stopHoldProgress: _stopHoldProgress,
              horizontal: true,
              onStart: _start,
              onStopHoldStart: _beginStopHold,
              onStopHoldCancel: _cancelStopHold,
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: _mapExpanded ? 300 : 190,
            child: _MapInset(
              points: _controller.track,
              expanded: _mapExpanded,
              onToggle: () => setState(() => _mapExpanded = !_mapExpanded),
            ),
          ),
        ],
      ),
    );
  }
}

class _PreflightReadiness {
  const _PreflightReadiness({
    required this.cameraReady,
    required this.gpsReady,
    required this.storageReady,
    required this.calibrationSet,
    required this.storageTimeLeft,
    required this.storageDetail,
    required this.gpsAccuracyM,
    required this.gpsFixUtc,
  });

  final bool cameraReady;
  final bool gpsReady;
  final bool storageReady;
  final bool calibrationSet;
  final Duration? storageTimeLeft;
  final String storageDetail;
  final double? gpsAccuracyM;
  final DateTime? gpsFixUtc;

  bool get canStart => cameraReady && gpsReady && storageReady;
}

class _TopStrip extends StatelessWidget {
  const _TopStrip({
    required this.gpsAccuracyM,
    required this.satelliteCount,
    required this.elapsed,
    required this.storageTimeLeft,
    required this.gpsReady,
  });

  final double? gpsAccuracyM;
  final int? satelliteCount;
  final Duration elapsed;
  final Duration? storageTimeLeft;
  final bool gpsReady;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textStyle = theme.textTheme.titleMedium?.copyWith(
      color: Colors.white,
      fontWeight: FontWeight.w900,
    );
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(178),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withAlpha(42)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        child: Row(
          children: [
            _StatusDot(color: gpsReady ? _readyColor : _warningColor),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                '${_gpsFixText(gpsAccuracyM)}  ${_satelliteText(satelliteCount)}',
                style: textStyle,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 12),
            Icon(Icons.timer_outlined, color: Colors.white, size: 22),
            const SizedBox(width: 6),
            Text(_formatDuration(elapsed), style: textStyle),
            const SizedBox(width: 12),
            Icon(Icons.sd_storage_outlined, color: Colors.white, size: 22),
            const SizedBox(width: 6),
            Text(_formatTimeLeft(storageTimeLeft), style: textStyle),
          ],
        ),
      ),
    );
  }
}

class _StatePill extends StatelessWidget {
  const _StatePill({required this.state, required this.animation});

  final _SurveyVisualState state;
  final Animation<double> animation;

  @override
  Widget build(BuildContext context) {
    final data = _StatePillData.forState(state);
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        final pulse = state == _SurveyVisualState.recording
            ? 0.75 + (animation.value * 0.25)
            : 1.0;
        return Transform.scale(
          scale: pulse,
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: data.color.withAlpha(232),
              borderRadius: BorderRadius.circular(999),
              boxShadow: [
                BoxShadow(
                  color: data.color.withAlpha(100),
                  blurRadius: state == _SurveyVisualState.recording ? 26 : 12,
                  spreadRadius: state == _SurveyVisualState.recording ? 4 : 0,
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(data.icon, color: Colors.white, size: 22),
                  const SizedBox(width: 8),
                  Text(
                    data.label,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _StatePillData {
  const _StatePillData({
    required this.label,
    required this.color,
    required this.icon,
  });

  final String label;
  final Color color;
  final IconData icon;

  factory _StatePillData.forState(_SurveyVisualState state) {
    switch (state) {
      case _SurveyVisualState.recording:
        return const _StatePillData(
          label: 'REC',
          color: _recordingColor,
          icon: Icons.fiber_manual_record,
        );
      case _SurveyVisualState.paused:
        return const _StatePillData(
          label: 'PAUSED - stationary',
          color: _warningColor,
          icon: Icons.pause,
        );
      case _SurveyVisualState.ready:
        return const _StatePillData(
          label: 'READY',
          color: _readyColor,
          icon: Icons.check_circle,
        );
      case _SurveyVisualState.stopping:
        return const _StatePillData(
          label: 'STOPPING',
          color: _idleColor,
          icon: Icons.stop_circle_outlined,
        );
      case _SurveyVisualState.idle:
        return const _StatePillData(
          label: 'IDLE',
          color: _idleColor,
          icon: Icons.radio_button_unchecked,
        );
    }
  }
}

class _SpeedReadout extends StatelessWidget {
  const _SpeedReadout({
    required this.speedMps,
    required this.units,
    this.compact = false,
  });

  final double speedMps;
  final UnitSystem units;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final speed = _speedValue(speedMps, units);
    final unit = units == UnitSystem.imperial ? 'mph' : 'km/h';
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(compact ? 220 : 150),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withAlpha(36)),
      ),
      child: Padding(
        padding: EdgeInsets.symmetric(
          horizontal: compact ? 18 : 24,
          vertical: compact ? 14 : 18,
        ),
        child: Row(
          mainAxisSize: compact ? MainAxisSize.max : MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              speed,
              style: theme.textTheme.displayLarge?.copyWith(
                color: Colors.white,
                fontSize: compact ? 68 : 96,
                fontWeight: FontWeight.w900,
                height: 0.88,
              ),
            ),
            const SizedBox(width: 10),
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                unit,
                style: theme.textTheme.headlineSmall?.copyWith(
                  color: Colors.white.withAlpha(224),
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ThumbRail extends StatelessWidget {
  const _ThumbRail({
    required this.controller,
    required this.settings,
    required this.freeBytes,
    required this.storageTarget,
    required this.stopHoldProgress,
    required this.onStart,
    required this.onStopHoldStart,
    required this.onStopHoldCancel,
    this.horizontal = false,
  });

  final CaptureSessionController controller;
  final AppSettings settings;
  final int? freeBytes;
  final StorageTarget? storageTarget;
  final double stopHoldProgress;
  final VoidCallback onStart;
  final VoidCallback onStopHoldStart;
  final void Function({bool resetProgress}) onStopHoldCancel;
  final bool horizontal;

  @override
  Widget build(BuildContext context) {
    final content = [
      _RecordControlButton(
        controller: controller,
        stopHoldProgress: stopHoldProgress,
        onStart: onStart,
        onStopHoldStart: onStopHoldStart,
        onStopHoldCancel: onStopHoldCancel,
      ),
      _RailMetric(
        icon: Icons.image_outlined,
        label: 'Frames',
        value: controller.estimatedFrameCount.toString(),
      ),
      const _RailMetric(
        icon: Icons.movie_creation_outlined,
        label: 'Segments',
        value: '1',
      ),
      _RailMetric(
        icon: Icons.my_location,
        label: 'GPS tick',
        value: controller.gpsSamples.toString(),
        active: controller.gpsSamples > 0,
      ),
      _RailMetric(
        icon: Icons.sensors,
        label: 'IMU tick',
        value: controller.imuSamples.toString(),
        active: controller.imuSamples > 0,
      ),
      _RailMetric(
        icon: Icons.sd_storage_outlined,
        label: storageTarget?.label ?? 'Storage',
        value: _formatTimeLeft(_estimatedTimeLeft(freeBytes, settings)),
      ),
    ];

    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(172),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withAlpha(42)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: horizontal
            ? ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: content.length,
                separatorBuilder: (context, index) => const SizedBox(width: 8),
                itemBuilder: (context, index) {
                  return SizedBox(
                    width: index == 0 ? 112 : 116,
                    child: content[index],
                  );
                },
              )
            : Column(
                children: [
                  content.first,
                  const SizedBox(height: 10),
                  Expanded(
                    child: DefaultTextStyle.merge(
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: Colors.white,
                      ),
                      child: ListView.separated(
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: content.length - 1,
                        separatorBuilder: (context, index) =>
                            const SizedBox(height: 8),
                        itemBuilder: (context, index) => content[index + 1],
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

class _RecordControlButton extends StatelessWidget {
  const _RecordControlButton({
    required this.controller,
    required this.stopHoldProgress,
    required this.onStart,
    required this.onStopHoldStart,
    required this.onStopHoldCancel,
  });

  final CaptureSessionController controller;
  final double stopHoldProgress;
  final VoidCallback onStart;
  final VoidCallback onStopHoldStart;
  final void Function({bool resetProgress}) onStopHoldCancel;

  @override
  Widget build(BuildContext context) {
    final isStop = controller.isRecording || controller.isStopping;
    final color = isStop ? _recordingColor : _readyColor;
    final label = controller.isStopping
        ? 'STOPPING'
        : (isStop ? 'HOLD' : 'START');
    final icon = isStop ? Icons.stop : Icons.fiber_manual_record;

    return GestureDetector(
      onTap: isStop
          ? () {
              unawaited(HapticFeedback.selectionClick());
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Hold Stop to end recording.')),
              );
            }
          : onStart,
      onLongPressStart: isStop ? (_) => onStopHoldStart() : null,
      onLongPressEnd: isStop
          ? (_) => onStopHoldCancel(resetProgress: true)
          : null,
      onLongPressCancel: isStop
          ? () => onStopHoldCancel(resetProgress: true)
          : null,
      child: Semantics(
        button: true,
        label: isStop ? 'Hold to stop recording' : 'Start survey',
        child: SizedBox(
          height: 104,
          child: Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                width: 98,
                height: 98,
                child: CircularProgressIndicator(
                  value: isStop ? stopHoldProgress.clamp(0.0, 1.0) : 1,
                  strokeWidth: 6,
                  color: Colors.white,
                  backgroundColor: Colors.white.withAlpha(55),
                ),
              ),
              Container(
                width: 86,
                height: 86,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: color,
                  boxShadow: [
                    BoxShadow(
                      color: color.withAlpha(118),
                      blurRadius: 24,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(icon, color: Colors.white, size: 28),
                    const SizedBox(height: 3),
                    Text(
                      label,
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w900,
                        fontSize: 12,
                      ),
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

class _RailMetric extends StatelessWidget {
  const _RailMetric({
    required this.icon,
    required this.label,
    required this.value,
    this.active = false,
  });

  final IconData icon;
  final String label;
  final String value;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white.withAlpha(28),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withAlpha(35)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: Row(
          children: [
            Stack(
              clipBehavior: Clip.none,
              children: [
                Icon(icon, color: Colors.white, size: 22),
                if (active)
                  const Positioned(
                    right: -3,
                    top: -3,
                    child: _StatusDot(color: _readyColor, size: 8),
                  ),
              ],
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    label,
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: Colors.white.withAlpha(205),
                      fontWeight: FontWeight.w700,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text(
                    value,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PreflightSheet extends StatelessWidget {
  const _PreflightSheet({
    required this.readiness,
    required this.checking,
    required this.onCheck,
    required this.onStart,
  });

  final _PreflightReadiness readiness;
  final bool checking;
  final VoidCallback onCheck;
  final VoidCallback? onStart;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withAlpha(242),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Pre-flight readiness',
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w900,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _ReadinessItem(
                  label: 'Camera ready',
                  detail: readiness.cameraReady ? 'Ready' : 'Check required',
                  state: readiness.cameraReady
                      ? _ReadinessState.good
                      : _ReadinessState.bad,
                ),
                _ReadinessItem(
                  label: 'GPS fix acquired',
                  detail: _gpsFixText(readiness.gpsAccuracyM),
                  state: readiness.gpsReady
                      ? _ReadinessState.good
                      : _ReadinessState.bad,
                ),
                _ReadinessItem(
                  label: 'Storage sufficient',
                  detail: readiness.storageDetail,
                  state: readiness.storageReady
                      ? _ReadinessState.good
                      : _ReadinessState.bad,
                ),
                _ReadinessItem(
                  label: 'Mount calibration',
                  detail: readiness.calibrationSet ? 'Set' : 'Set later',
                  state: readiness.calibrationSet
                      ? _ReadinessState.good
                      : _ReadinessState.warning,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: checking ? null : onCheck,
                    icon: checking
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.fact_check_outlined),
                    label: Text(checking ? 'Checking' : 'Check'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  flex: 2,
                  child: FilledButton.icon(
                    onPressed: onStart,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('START SURVEY'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

enum _ReadinessState { good, warning, bad }

class _ReadinessItem extends StatelessWidget {
  const _ReadinessItem({
    required this.label,
    required this.detail,
    required this.state,
  });

  final String label;
  final String detail;
  final _ReadinessState state;

  @override
  Widget build(BuildContext context) {
    final color = switch (state) {
      _ReadinessState.good => _readyColor,
      _ReadinessState.warning => _warningColor,
      _ReadinessState.bad => _recordingColor,
    };
    final icon = switch (state) {
      _ReadinessState.good => Icons.check_circle,
      _ReadinessState.warning => Icons.warning_amber_rounded,
      _ReadinessState.bad => Icons.cancel,
    };
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 190, minHeight: 64),
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: color.withAlpha(30),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withAlpha(170)),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: color, size: 24),
              const SizedBox(width: 8),
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      label,
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        fontWeight: FontWeight.w900,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      detail,
                      style: Theme.of(context).textTheme.labelMedium,
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

class _MapInset extends StatelessWidget {
  const _MapInset({
    required this.points,
    required this.expanded,
    required this.onToggle,
  });

  final List<TrackPoint> points;
  final bool expanded;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onToggle,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.black.withAlpha(150),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withAlpha(52)),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Stack(
            fit: StackFit.expand,
            children: [
              _LiveMap(points: points),
              Positioned(
                right: 8,
                top: 8,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.black.withAlpha(170),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      expanded ? Icons.close_fullscreen : Icons.open_in_full,
                      color: Colors.white,
                      size: 22,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
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
      borderRadius: BorderRadius.circular(fill ? 0 : 8),
      child: Container(
        color: Colors.black,
        alignment: Alignment.center,
        child: active != null && active.value.isInitialized
            ? _CameraPreviewContent(controller: active, fill: fill)
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.videocam_off,
                    color: Colors.white54,
                    size: 56,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Camera waiting',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: Colors.white70,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ],
              ),
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
    return FlutterMap(
      key: ValueKey('${latLngs.length}_${center.latitude}_${center.longitude}'),
      options: MapOptions(
        initialCenter: center,
        initialZoom: latLngs.isEmpty ? 2 : 16,
        interactionOptions: const InteractionOptions(
          flags: InteractiveFlag.none,
        ),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.qualityunit.roadsurvey_recorder',
        ),
        if (latLngs.length > 1)
          PolylineLayer(
            polylines: [
              Polyline(points: latLngs, strokeWidth: 5, color: _readyColor),
            ],
          ),
        if (latLngs.isNotEmpty)
          MarkerLayer(
            markers: [
              Marker(
                point: latLngs.last,
                width: 38,
                height: 38,
                child: const Icon(Icons.navigation, color: _recordingColor),
              ),
            ],
          ),
      ],
    );
  }
}

class _HudScrim extends StatelessWidget {
  const _HudScrim();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.black.withAlpha(132),
            Colors.black.withAlpha(18),
            Colors.black.withAlpha(128),
          ],
        ),
      ),
    );
  }
}

class _DimOverlay extends StatelessWidget {
  const _DimOverlay();

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: DecoratedBox(
        decoration: BoxDecoration(color: Colors.black.withAlpha(105)),
      ),
    );
  }
}

class _MessageStack extends StatelessWidget {
  const _MessageStack({required this.warning, required this.error});

  final String? warning;
  final String? error;

  @override
  Widget build(BuildContext context) {
    final messages = [
      if (warning != null)
        _Banner(
          icon: Icons.info_outline,
          color: _warningColor,
          message: warning!,
        ),
      if (error != null)
        _Banner(
          icon: Icons.error_outline,
          color: _recordingColor,
          message: error!,
        ),
    ];
    if (messages.isEmpty) {
      return const SizedBox.shrink();
    }
    return Column(mainAxisSize: MainAxisSize.min, children: messages);
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
          color: color.withAlpha(238),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Icon(icon, size: 24, color: Colors.white),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  message,
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusDot extends StatelessWidget {
  const _StatusDot({required this.color, this.size = 12});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
        boxShadow: [BoxShadow(color: color.withAlpha(150), blurRadius: 8)],
      ),
    );
  }
}

String _speedValue(double speedMps, UnitSystem units) {
  final value = units == UnitSystem.imperial
      ? speedMps * 2.236936
      : speedMps * 3.6;
  if (value >= 100) {
    return value.toStringAsFixed(0);
  }
  return value.toStringAsFixed(1);
}

String _gpsFixText(double? accuracy) {
  if (accuracy == null) {
    return 'GPS no fix';
  }
  return 'GPS ${accuracy.toStringAsFixed(1)} m';
}

String _satelliteText(int? satellites) {
  if (satellites == null) {
    return '-- sat';
  }
  return '$satellites sat';
}

String _formatDuration(Duration duration) {
  final hours = duration.inHours;
  final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
  final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
  return '$hours:$minutes:$seconds';
}

Duration? _estimatedTimeLeft(int? bytes, AppSettings settings) {
  if (bytes == null || bytes <= 0) {
    return null;
  }
  final bytesPerSecond = _estimatedBitrateBps(settings) / 8;
  if (bytesPerSecond <= 0) {
    return null;
  }
  return Duration(seconds: (bytes / bytesPerSecond).floor());
}

double _estimatedBitrateBps(AppSettings settings) {
  final base30Fps = switch (settings.resolution) {
    CaptureResolution.p720 => 6000000.0,
    CaptureResolution.p1080 => 12000000.0,
    CaptureResolution.p2160 => 50000000.0,
    CaptureResolution.max => 60000000.0,
  };
  final codecFactor = settings.codec == CaptureCodec.hevc ? 0.65 : 1.0;
  final fpsFactor = settings.effectiveContinuousFps / 30.0;
  return base30Fps * codecFactor * fpsFactor;
}

String _formatTimeLeft(Duration? duration) {
  if (duration == null) {
    return 'Storage unknown';
  }
  if (duration.inMinutes < 1) {
    return '<1 min';
  }
  if (duration.inHours >= 1) {
    return '~${(duration.inMinutes / 60).toStringAsFixed(1)} h';
  }
  return '~${duration.inMinutes} min';
}

String _storageDetail(StorageTarget? target, Duration? duration) {
  final timeLeft = _formatTimeLeft(duration);
  if (target == null) {
    return timeLeft;
  }
  if (target.externalUnavailable) {
    return 'External missing; Internal $timeLeft';
  }
  return '${target.label} $timeLeft';
}
