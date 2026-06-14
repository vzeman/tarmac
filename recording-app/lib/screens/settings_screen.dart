import 'package:flutter/material.dart';

import '../settings/app_settings.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({
    super.key,
    required this.settings,
    required this.onChanged,
  });

  final AppSettings settings;
  final Future<void> Function(AppSettings settings) onChanged;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        _SliderTile(
          title: 'Frame spacing',
          value: settings.frameSpacingM,
          min: 1,
          max: 20,
          divisions: 19,
          suffix: 'm',
          onChanged: (value) =>
              onChanged(settings.copyWith(frameSpacingM: value)),
        ),
        _IntSliderTile(
          title: 'Max FPS',
          value: settings.maxFps,
          min: 1,
          max: 30,
          onChanged: (value) => onChanged(settings.copyWith(maxFps: value)),
        ),
        _IntSliderTile(
          title: 'Min FPS',
          value: settings.minFps,
          min: 1,
          max: 8,
          onChanged: (value) => onChanged(settings.copyWith(minFps: value)),
        ),
        _SliderTile(
          title: 'Pause speed',
          value: settings.pauseSpeedKmh,
          min: 0,
          max: 15,
          divisions: 30,
          suffix: 'km/h',
          onChanged: (value) =>
              onChanged(settings.copyWith(pauseSpeedKmh: value)),
        ),
        _IntSliderTile(
          title: 'Pause debounce',
          value: settings.pauseDebounceS,
          min: 1,
          max: 15,
          suffix: 's',
          onChanged: (value) =>
              onChanged(settings.copyWith(pauseDebounceS: value)),
        ),
        _DropdownTile<CaptureMode>(
          title: 'Capture mode',
          value: settings.captureMode,
          values: CaptureMode.values,
          labelFor: (value) => value.label,
          onChanged: (value) =>
              onChanged(settings.copyWith(captureMode: value)),
        ),
        _DropdownTile<CaptureResolution>(
          title: 'Resolution',
          value: settings.resolution,
          values: CaptureResolution.values,
          labelFor: (value) => value.label,
          onChanged: (value) => onChanged(settings.copyWith(resolution: value)),
        ),
        _DropdownTile<CaptureCodec>(
          title: 'Codec',
          value: settings.codec,
          values: CaptureCodec.values,
          labelFor: (value) => value.label,
          onChanged: (value) => onChanged(settings.copyWith(codec: value)),
        ),
        _SliderTile(
          title: 'Max segment size',
          value: settings.maxSegmentGb,
          min: 1,
          max: 50,
          divisions: 49,
          suffix: 'GB',
          onChanged: (value) =>
              onChanged(settings.copyWith(maxSegmentGb: value)),
        ),
        _DropdownTile<StorageLocation>(
          title: 'Storage',
          value: settings.storageLocation,
          values: StorageLocation.values,
          labelFor: (value) => value.label,
          onChanged: (value) =>
              onChanged(settings.copyWith(storageLocation: value)),
        ),
        SwitchListTile(
          title: const Text('Keep screen on'),
          value: settings.keepScreenOn,
          onChanged: (value) =>
              onChanged(settings.copyWith(keepScreenOn: value)),
        ),
        _DropdownTile<UnitSystem>(
          title: 'Units',
          value: settings.units,
          values: UnitSystem.values,
          labelFor: (value) => value.label,
          onChanged: (value) => onChanged(settings.copyWith(units: value)),
        ),
        const SizedBox(height: 12),
        const _TodoTile(
          title: 'Adaptive distance capture',
          milestone: 'SPEC M3',
        ),
        const _TodoTile(title: 'Stationary auto-pause', milestone: 'SPEC M3'),
        const _TodoTile(
          title: 'External USB/SAF storage',
          milestone: 'SPEC M4',
        ),
        const _TodoTile(title: 'Segment auto-split', milestone: 'SPEC M4'),
        const _TodoTile(
          title: 'Background recording and thermal policy',
          milestone: 'SPEC M7',
        ),
        const _TodoTile(
          title: 'Tarmac sidecar ingestion',
          milestone: 'SPEC M6',
        ),
      ],
    );
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
    return ListTile(
      title: Text(title),
      subtitle: Slider(
        value: value.clamp(min, max),
        min: min,
        max: max,
        divisions: divisions,
        label: '${value.toStringAsFixed(1)} $suffix',
        onChanged: onChanged,
      ),
      trailing: SizedBox(
        width: 72,
        child: Text(
          '${value.toStringAsFixed(value.truncateToDouble() == value ? 0 : 1)} $suffix',
          textAlign: TextAlign.end,
        ),
      ),
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
    return ListTile(
      title: Text(title),
      subtitle: Slider(
        value: value.clamp(min, max).toDouble(),
        min: min.toDouble(),
        max: max.toDouble(),
        divisions: max - min,
        label: suffix.isEmpty ? '$value' : '$value $suffix',
        onChanged: (value) => onChanged(value.round()),
      ),
      trailing: SizedBox(
        width: 72,
        child: Text(
          suffix.isEmpty ? '$value' : '$value $suffix',
          textAlign: TextAlign.end,
        ),
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
  });

  final String title;
  final T value;
  final List<T> values;
  final String Function(T value) labelFor;
  final ValueChanged<T> onChanged;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(title),
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

class _TodoTile extends StatelessWidget {
  const _TodoTile({required this.title, required this.milestone});

  final String title;
  final String milestone;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: const Icon(Icons.pending_actions),
      title: Text(title),
      trailing: Text(milestone),
    );
  }
}
