import 'package:flutter/material.dart';

import '../services/storage_service.dart';
import '../settings/app_settings.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({
    super.key,
    required this.settings,
    required this.onChanged,
    required this.storageService,
  });

  final AppSettings settings;
  final Future<void> Function(AppSettings settings) onChanged;
  final StorageService storageService;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final horizontalPadding = constraints.maxWidth >= 700 ? 28.0 : 12.0;
        return SafeArea(
          child: ListView(
            padding: EdgeInsets.fromLTRB(
              horizontalPadding,
              16,
              horizontalPadding,
              28,
            ),
            children: [
              _SettingsSection(
                title: 'Capture',
                children: [
                  _DropdownTile<CaptureResolution>(
                    title: 'Resolution',
                    subtitle: 'Camera capture preset',
                    value: settings.resolution,
                    values: _resolutionChoices(settings.resolution),
                    labelFor: (value) => value.label,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(resolution: value)),
                  ),
                  _SegmentedIntTile(
                    title: 'FPS',
                    value: settings.maxFps,
                    values: const [30, 60],
                    suffix: ' fps',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(maxFps: value)),
                  ),
                  SwitchListTile(
                    title: const Text('HEVC'),
                    subtitle: const Text('Use HEVC when supported'),
                    value: settings.codec == CaptureCodec.hevc,
                    onChanged: (value) => onChanged(
                      settings.copyWith(
                        codec: value ? CaptureCodec.hevc : CaptureCodec.h264,
                      ),
                    ),
                  ),
                  _SliderTile(
                    title: 'Segment size',
                    value: settings.maxSegmentGb,
                    min: 1,
                    max: 50,
                    divisions: 49,
                    suffix: 'GB',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(maxSegmentGb: value)),
                  ),
                ],
              ),
              _SettingsSection(
                title: 'Survey',
                children: [
                  _SliderTile(
                    title: 'Downstream frame spacing',
                    value: settings.frameSpacingM,
                    min: 1,
                    max: 20,
                    divisions: 19,
                    suffix: 'm',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(frameSpacingM: value)),
                  ),
                  SwitchListTile(
                    title: const Text('Auto-pause while stationary'),
                    subtitle: const Text('Finalize and resume video segments'),
                    value: settings.autoPauseEnabled,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(autoPauseEnabled: value)),
                  ),
                  _SliderTile(
                    title: 'Auto-pause speed',
                    value: settings.pauseSpeedKmh,
                    min: 0,
                    max: 15,
                    divisions: 30,
                    suffix: 'km/h',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(pauseSpeedKmh: value)),
                  ),
                  _IntSliderTile(
                    title: 'Auto-pause debounce',
                    value: settings.pauseDebounceS,
                    min: 1,
                    max: 15,
                    suffix: 's',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(pauseDebounceS: value)),
                  ),
                  _IntSliderTile(
                    title: 'Resume sensitivity',
                    value: settings.resumeSensitivity,
                    min: 1,
                    max: 10,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(resumeSensitivity: value)),
                  ),
                ],
              ),
              _SettingsSection(
                title: 'Calibration',
                children: [
                  SwitchListTile(
                    title: const Text('Mount calibration set'),
                    subtitle: const Text('Required for true-scale measuring'),
                    value: settings.mountCalibrationSet,
                    onChanged: (value) => onChanged(
                      settings.copyWith(mountCalibrationSet: value),
                    ),
                  ),
                  _SliderTile(
                    title: 'Mount height',
                    value: settings.mountHeightM,
                    min: 0.3,
                    max: 3.0,
                    divisions: 27,
                    suffix: 'm',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(mountHeightM: value)),
                  ),
                  _SliderTile(
                    title: 'Mount tilt',
                    value: settings.mountTiltDeg,
                    min: -30,
                    max: 30,
                    divisions: 60,
                    suffix: 'deg',
                    onChanged: (value) =>
                        onChanged(settings.copyWith(mountTiltDeg: value)),
                  ),
                  _DropdownTile<LensProfile>(
                    title: 'Lens',
                    subtitle: 'Camera profile for scale metadata',
                    value: settings.lensProfile,
                    values: LensProfile.values,
                    labelFor: (value) => value.label,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(lensProfile: value)),
                  ),
                ],
              ),
              _SettingsSection(
                title: 'Storage',
                children: [
                  _DropdownTile<StorageLocation>(
                    title: 'Location',
                    subtitle: 'Final session destination',
                    value: settings.storageLocation,
                    values: const [
                      StorageLocation.internal,
                      StorageLocation.external,
                    ],
                    labelFor: (value) => value.label,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(storageLocation: value)),
                  ),
                  _ExternalLocationTile(
                    settings: settings,
                    storageService: storageService,
                    onChanged: onChanged,
                  ),
                ],
              ),
              _SettingsSection(
                title: 'Display',
                children: [
                  _DropdownTile<DisplayTheme>(
                    title: 'Theme',
                    value: settings.displayTheme,
                    values: DisplayTheme.values,
                    labelFor: (value) => value.label,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(displayTheme: value)),
                  ),
                  _DropdownTile<UnitSystem>(
                    title: 'Units',
                    value: settings.units,
                    values: UnitSystem.values,
                    labelFor: (value) => value.label,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(units: value)),
                  ),
                  SwitchListTile(
                    title: const Text('Keep awake'),
                    subtitle: const Text('Prevent sleep while recording'),
                    value: settings.keepScreenOn,
                    onChanged: (value) =>
                        onChanged(settings.copyWith(keepScreenOn: value)),
                  ),
                  SwitchListTile(
                    title: const Text('Auto dim while recording'),
                    subtitle: const Text('Dim the overlay, keep recording'),
                    value: settings.autoDimWhileRecording,
                    onChanged: (value) => onChanged(
                      settings.copyWith(autoDimWhileRecording: value),
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  List<CaptureResolution> _resolutionChoices(CaptureResolution current) {
    final values = <CaptureResolution>{
      CaptureResolution.p1080,
      CaptureResolution.p2160,
      current,
    }.toList();
    values.sort((a, b) => a.index.compareTo(b.index));
    return values;
  }
}

class _SettingsSection extends StatelessWidget {
  const _SettingsSection({required this.title, required this.children});

  final String title;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
            child: Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w900,
                color: theme.colorScheme.primary,
              ),
            ),
          ),
          DecoratedBox(
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: theme.colorScheme.outlineVariant),
            ),
            child: Column(
              children: [
                for (var index = 0; index < children.length; index += 1) ...[
                  children[index],
                  if (index != children.length - 1)
                    Divider(
                      height: 1,
                      indent: 18,
                      endIndent: 18,
                      color: theme.colorScheme.outlineVariant,
                    ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ExternalLocationTile extends StatefulWidget {
  const _ExternalLocationTile({
    required this.settings,
    required this.storageService,
    required this.onChanged,
  });

  final AppSettings settings;
  final StorageService storageService;
  final Future<void> Function(AppSettings settings) onChanged;

  @override
  State<_ExternalLocationTile> createState() => _ExternalLocationTileState();
}

class _ExternalLocationTileState extends State<_ExternalLocationTile> {
  bool? _available;
  bool _choosing = false;
  bool _checking = false;

  @override
  void initState() {
    super.initState();
    _refreshAvailability();
  }

  Future<void> _refreshAvailability() async {
    if (_checking) {
      return;
    }
    setState(() => _checking = true);
    final available = await widget.storageService.externalAvailable();
    if (!mounted) {
      return;
    }
    setState(() {
      _available = available;
      _checking = false;
    });
  }

  Future<void> _chooseExternal() async {
    if (_choosing) {
      return;
    }
    setState(() => _choosing = true);
    final selected = await widget.storageService.chooseExternal();
    final available = await widget.storageService.externalAvailable();
    if (!mounted) {
      return;
    }
    setState(() {
      _available = available;
      _choosing = false;
    });

    final messenger = ScaffoldMessenger.of(context);
    if (!selected) {
      messenger.showSnackBar(
        const SnackBar(content: Text('No external location selected.')),
      );
      return;
    }

    await widget.onChanged(
      widget.settings.copyWith(storageLocation: StorageLocation.external),
    );
    if (!mounted) {
      return;
    }
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          available
              ? 'External location selected.'
              : 'External location saved, but it is not reachable now.',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: const Icon(Icons.drive_folder_upload_outlined),
      title: const Text('Choose external location'),
      subtitle: Text(_statusText()),
      trailing: _choosing || _checking
          ? const SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : Icon(
              _available == true
                  ? Icons.check_circle_outline
                  : Icons.error_outline,
            ),
      onTap: _choosing ? null : _chooseExternal,
    );
  }

  String _statusText() {
    if (_choosing) {
      return 'Opening system folder picker';
    }
    if (_checking) {
      return 'Checking external access';
    }
    if (_available == true) {
      return 'External location granted and reachable';
    }
    if (widget.settings.storageLocation == StorageLocation.external) {
      return 'External is selected, but no location is reachable';
    }
    return 'Grant access before recording to external storage';
  }
}

class _SliderTile extends StatelessWidget {
  const _SliderTile({
    required this.title,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.suffix,
    required this.onChanged,
  });

  final String title;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final String suffix;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    final display = _formatNumber(value);
    return ListTile(
      minVerticalPadding: 12,
      title: Text(title),
      subtitle: Slider(
        value: value.clamp(min, max),
        min: min,
        max: max,
        divisions: divisions,
        label: '$display $suffix',
        onChanged: onChanged,
      ),
      trailing: _ValueChip(value: '$display $suffix'),
    );
  }
}

class _IntSliderTile extends StatelessWidget {
  const _IntSliderTile({
    required this.title,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
    this.suffix = '',
  });

  final String title;
  final int value;
  final int min;
  final int max;
  final String suffix;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final display = suffix.isEmpty ? '$value' : '$value $suffix';
    return ListTile(
      minVerticalPadding: 12,
      title: Text(title),
      subtitle: Slider(
        value: value.clamp(min, max).toDouble(),
        min: min.toDouble(),
        max: max.toDouble(),
        divisions: max - min,
        label: display,
        onChanged: (value) => onChanged(value.round()),
      ),
      trailing: _ValueChip(value: display),
    );
  }
}

class _SegmentedIntTile extends StatelessWidget {
  const _SegmentedIntTile({
    required this.title,
    required this.value,
    required this.values,
    required this.suffix,
    required this.onChanged,
  });

  final String title;
  final int value;
  final List<int> values;
  final String suffix;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final selected = values.contains(value) ? value : values.first;
    return ListTile(
      title: Text(title),
      trailing: SegmentedButton<int>(
        segments: [
          for (final option in values)
            ButtonSegment<int>(value: option, label: Text('$option$suffix')),
        ],
        selected: {selected},
        showSelectedIcon: false,
        onSelectionChanged: (selected) => onChanged(selected.single),
      ),
    );
  }
}

class _DropdownTile<T> extends StatelessWidget {
  const _DropdownTile({
    required this.title,
    required this.value,
    required this.values,
    required this.labelFor,
    required this.onChanged,
    this.subtitle,
  });

  final String title;
  final String? subtitle;
  final T value;
  final List<T> values;
  final String Function(T value) labelFor;
  final ValueChanged<T> onChanged;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(title),
      subtitle: subtitle == null ? null : Text(subtitle!),
      trailing: DropdownButton<T>(
        value: value,
        items: values
            .map(
              (item) =>
                  DropdownMenuItem<T>(value: item, child: Text(labelFor(item))),
            )
            .toList(),
        onChanged: (value) {
          if (value != null) {
            onChanged(value);
          }
        },
      ),
    );
  }
}

class _ValueChip extends StatelessWidget {
  const _ValueChip({required this.value});

  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      constraints: const BoxConstraints(minWidth: 78),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        value,
        textAlign: TextAlign.center,
        style: theme.textTheme.labelLarge?.copyWith(
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

String _formatNumber(double value) {
  return value.truncateToDouble() == value
      ? value.toStringAsFixed(0)
      : value.toStringAsFixed(1);
}
