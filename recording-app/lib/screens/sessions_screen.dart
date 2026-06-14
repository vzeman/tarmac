import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import '../services/session_repository.dart';
import 'session_video_player_screen.dart';

class SessionsScreen extends StatefulWidget {
  const SessionsScreen({
    super.key,
    required this.sessions,
    required this.sessionRepository,
    required this.onRefresh,
  });

  final List<SessionSummary> sessions;
  final SessionRepository sessionRepository;
  final Future<void> Function() onRefresh;

  @override
  State<SessionsScreen> createState() => _SessionsScreenState();
}

class _SessionsScreenState extends State<SessionsScreen> {
  final Set<String> _selectedIds = {};
  bool _selectionMode = false;
  bool _deleting = false;

  @override
  void didUpdateWidget(covariant SessionsScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    final visibleIds = widget.sessions.map((session) => session.id).toSet();
    _selectedIds.removeWhere((id) => !visibleIds.contains(id));
    if (_selectedIds.isEmpty && _selectionMode && widget.sessions.isEmpty) {
      _selectionMode = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final horizontalPadding = constraints.maxWidth >= 700 ? 28.0 : 12.0;
        final padding = EdgeInsets.fromLTRB(
          horizontalPadding,
          16,
          horizontalPadding,
          28,
        );

        if (widget.sessions.isEmpty) {
          return SafeArea(
            child: RefreshIndicator(
              onRefresh: widget.onRefresh,
              child: ListView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: padding,
                children: const [
                  SizedBox(height: 160),
                  Center(child: Text('No recorded sessions')),
                ],
              ),
            ),
          );
        }

        return SafeArea(
          child: RefreshIndicator(
            onRefresh: widget.onRefresh,
            child: ListView.separated(
              padding: padding,
              itemBuilder: (context, index) {
                if (index == 0) {
                  return _SessionsHeader(
                    sessions: widget.sessions,
                    selectionMode: _selectionMode,
                    selectedCount: _selectedIds.length,
                    deleting: _deleting,
                    onToggleSelectionMode: _toggleSelectionMode,
                    onDeleteAll: () => _confirmAndDelete(widget.sessions),
                    onDeleteSelected: _selectedIds.isEmpty
                        ? null
                        : () => _confirmAndDelete(_selectedSessions()),
                  );
                }

                final session = widget.sessions[index - 1];
                return Dismissible(
                  key: ValueKey(session.id),
                  direction: _selectionMode
                      ? DismissDirection.none
                      : DismissDirection.endToStart,
                  confirmDismiss: (_) async {
                    return _confirmDeleteDialog(context, [session]);
                  },
                  onDismissed: (_) {
                    _deleteSessions([session]);
                  },
                  background: const _DeleteBackground(),
                  child: _SessionCard(
                    session: session,
                    sessionRepository: widget.sessionRepository,
                    selected: _selectedIds.contains(session.id),
                    selectionMode: _selectionMode,
                    onSelectionChanged: (selected) =>
                        _setSelected(session.id, selected),
                    onPlay: () => _openVideoPlayer(session),
                    onTap: () {
                      if (_selectionMode) {
                        _setSelected(
                          session.id,
                          !_selectedIds.contains(session.id),
                        );
                        return;
                      }
                      Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => SessionDetailScreen(
                            session: session,
                            sessionRepository: widget.sessionRepository,
                            onDeleted: widget.onRefresh,
                          ),
                        ),
                      );
                    },
                  ),
                );
              },
              separatorBuilder: (context, index) => const SizedBox(height: 12),
              itemCount: widget.sessions.length + 1,
            ),
          ),
        );
      },
    );
  }

  void _toggleSelectionMode() {
    setState(() {
      _selectionMode = !_selectionMode;
      _selectedIds.clear();
    });
  }

  void _setSelected(String id, bool selected) {
    setState(() {
      if (selected) {
        _selectedIds.add(id);
      } else {
        _selectedIds.remove(id);
      }
    });
  }

  List<SessionSummary> _selectedSessions() {
    return widget.sessions
        .where((session) => _selectedIds.contains(session.id))
        .toList();
  }

  void _openVideoPlayer(SessionSummary session) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => SessionVideoPlayerScreen(
          session: session,
          sessionRepository: widget.sessionRepository,
        ),
      ),
    );
  }

  Future<void> _confirmAndDelete(List<SessionSummary> sessions) async {
    final confirmed = await _confirmDeleteDialog(context, sessions);
    if (confirmed) {
      await _deleteSessions(sessions);
    }
  }

  Future<void> _deleteSessions(List<SessionSummary> sessions) async {
    if (_deleting || sessions.isEmpty) {
      return;
    }
    setState(() => _deleting = true);
    if (sessions.length == 1) {
      await widget.sessionRepository.deleteSession(sessions.single);
    } else {
      await widget.sessionRepository.deleteSessions(sessions);
    }
    await widget.onRefresh();
    if (!mounted) {
      return;
    }
    setState(() {
      _selectedIds.removeAll(sessions.map((session) => session.id));
      _selectionMode = _selectedIds.isNotEmpty;
      _deleting = false;
    });
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(_deleteSnackText(sessions.length))));
  }
}

class _SessionsHeader extends StatelessWidget {
  const _SessionsHeader({
    required this.sessions,
    required this.selectionMode,
    required this.selectedCount,
    required this.deleting,
    required this.onToggleSelectionMode,
    required this.onDeleteAll,
    required this.onDeleteSelected,
  });

  final List<SessionSummary> sessions;
  final bool selectionMode;
  final int selectedCount;
  final bool deleting;
  final VoidCallback onToggleSelectionMode;
  final VoidCallback onDeleteAll;
  final VoidCallback? onDeleteSelected;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final totalBytes = sessions.fold<int>(
      0,
      (sum, session) => sum + session.totalBytes,
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Sessions',
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  Text(
                    '${sessions.length} sessions  ${_formatBytes(totalBytes)}',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
            TextButton(
              onPressed: deleting ? null : onToggleSelectionMode,
              child: Text(selectionMode ? 'Done' : 'Select'),
            ),
            const SizedBox(width: 8),
            if (selectionMode)
              FilledButton.icon(
                onPressed: deleting ? null : onDeleteSelected,
                icon: const Icon(Icons.delete_outline),
                label: Text(
                  selectedCount == 0 ? 'Delete' : 'Delete $selectedCount',
                ),
              )
            else
              OutlinedButton.icon(
                onPressed: deleting ? null : onDeleteAll,
                icon: const Icon(Icons.delete_sweep_outlined),
                label: const Text('Delete all'),
              ),
          ],
        ),
        const SizedBox(height: 12),
      ],
    );
  }
}

class _SessionCard extends StatelessWidget {
  const _SessionCard({
    required this.session,
    required this.sessionRepository,
    required this.selected,
    required this.selectionMode,
    required this.onSelectionChanged,
    required this.onPlay,
    required this.onTap,
  });

  final SessionSummary session;
  final SessionRepository sessionRepository;
  final bool selected;
  final bool selectionMode;
  final ValueChanged<bool> onSelectionChanged;
  final VoidCallback onPlay;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (selectionMode) ...[
                Checkbox(
                  value: selected,
                  onChanged: (value) => onSelectionChanged(value ?? false),
                ),
                const SizedBox(width: 8),
              ],
              SizedBox(
                width: 150,
                height: 100,
                child: _SessionThumbnail(
                  session: session,
                  sessionRepository: sessionRepository,
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: FutureBuilder<List<TrackPoint>>(
                  future: sessionRepository.loadTrackPoints(session),
                  builder: (context, snapshot) {
                    final points = snapshot.data ?? const <TrackPoint>[];
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          DateFormat.yMMMd().add_Hm().format(
                            session.startedAtUtc.toLocal(),
                          ),
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.w900,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 6),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            _SessionFact(
                              icon: Icons.route_outlined,
                              value: _formatDistance(_trackDistance(points)),
                            ),
                            _SessionFact(
                              icon: Icons.timer_outlined,
                              value: _formatDuration(
                                Duration(milliseconds: session.durationMs),
                              ),
                            ),
                            _SessionFact(
                              icon: Icons.image_outlined,
                              value: '${session.frameCount} frames',
                            ),
                            const _SessionFact(
                              icon: Icons.movie_creation_outlined,
                              value: '1 segment',
                            ),
                            _SessionFact(
                              icon: Icons.storage,
                              value: _formatBytes(session.totalBytes),
                            ),
                            if (session.isExternal)
                              _SessionFact(
                                icon: session.storageAvailable
                                    ? Icons.usb
                                    : Icons.usb_off,
                                value: session.storageAvailable
                                    ? 'External'
                                    : 'External unavailable',
                              ),
                          ],
                        ),
                      ],
                    );
                  },
                ),
              ),
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (!selectionMode)
                    IconButton(
                      tooltip: 'Play video',
                      onPressed: onPlay,
                      icon: const Icon(Icons.play_circle_outline),
                    ),
                  const Icon(Icons.chevron_right),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class SessionDetailScreen extends StatelessWidget {
  const SessionDetailScreen({
    super.key,
    required this.session,
    required this.sessionRepository,
    required this.onDeleted,
  });

  final SessionSummary session;
  final SessionRepository sessionRepository;
  final Future<void> Function() onDeleted;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          DateFormat.yMMMd().add_Hm().format(session.startedAtUtc.toLocal()),
        ),
        actions: [
          IconButton(
            tooltip: 'Delete session',
            icon: const Icon(Icons.delete_outline),
            onPressed: () => _deleteFromDetail(context),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final horizontalPadding = constraints.maxWidth >= 700 ? 28.0 : 12.0;
            final mapHeight = constraints.maxWidth > constraints.maxHeight
                ? 250.0
                : 320.0;
            return FutureBuilder<List<TrackPoint>>(
              future: sessionRepository.loadTrackPoints(session),
              builder: (context, snapshot) {
                final points = snapshot.data ?? const <TrackPoint>[];
                return ListView(
                  padding: EdgeInsets.fromLTRB(
                    horizontalPadding,
                    14,
                    horizontalPadding,
                    28,
                  ),
                  children: [
                    SizedBox(
                      height: mapHeight,
                      child: _SessionMap(points: points),
                    ),
                    const SizedBox(height: 14),
                    Wrap(
                      spacing: 10,
                      runSpacing: 10,
                      children: [
                        _SummaryTile(
                          icon: Icons.route_outlined,
                          label: 'Distance',
                          value: _formatDistance(_trackDistance(points)),
                        ),
                        _SummaryTile(
                          icon: Icons.timer_outlined,
                          label: 'Duration',
                          value: _formatDuration(
                            Duration(milliseconds: session.durationMs),
                          ),
                        ),
                        _SummaryTile(
                          icon: Icons.image_outlined,
                          label: 'Frames',
                          value: session.frameCount.toString(),
                        ),
                        const _SummaryTile(
                          icon: Icons.movie_creation_outlined,
                          label: 'Segments',
                          value: '1',
                        ),
                        _SummaryTile(
                          icon: Icons.storage,
                          label: 'Size',
                          value: _formatBytes(session.totalBytes),
                        ),
                        _SummaryTile(
                          icon: session.isExternal
                              ? (session.storageAvailable
                                    ? Icons.usb
                                    : Icons.usb_off)
                              : Icons.phone_iphone,
                          label: 'Storage',
                          value: session.isExternal
                              ? (session.storageAvailable
                                    ? 'External'
                                    : 'Unavailable')
                              : 'Internal',
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    FilledButton.icon(
                      onPressed: () => _openVideoPlayer(context),
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Play video'),
                    ),
                    const SizedBox(height: 10),
                    OutlinedButton.icon(
                      onPressed: null,
                      icon: const Icon(Icons.ios_share_outlined),
                      label: const Text('Export to Tarmac'),
                    ),
                    const SizedBox(height: 10),
                    FilledButton.icon(
                      style: FilledButton.styleFrom(
                        backgroundColor: Theme.of(context).colorScheme.error,
                        foregroundColor: Theme.of(context).colorScheme.onError,
                      ),
                      onPressed: () => _deleteFromDetail(context),
                      icon: const Icon(Icons.delete_forever_outlined),
                      label: const Text('DELETE SESSION'),
                    ),
                    const SizedBox(height: 16),
                    _DetailRow(
                      label: 'Started',
                      value: session.startedAtUtc.toLocal().toString(),
                    ),
                    _DetailRow(
                      label: 'Ended',
                      value: session.endedAtUtc.toLocal().toString(),
                    ),
                    _DetailRow(
                      label: 'GPS samples',
                      value: session.gpsSampleCount.toString(),
                    ),
                    _DetailRow(
                      label: 'IMU samples',
                      value: session.imuSampleCount.toString(),
                    ),
                    _DetailRow(
                      label: 'Storage',
                      value: session.isExternal
                          ? (session.storageAvailable
                                ? 'External'
                                : 'External unavailable')
                          : 'Internal',
                    ),
                    _DetailRow(label: 'Video', value: session.videoPath),
                    _DetailRow(label: 'Sidecar', value: session.sidecarPath),
                    _DetailRow(label: 'GPX', value: session.gpxPath),
                  ],
                );
              },
            );
          },
        ),
      ),
    );
  }

  void _openVideoPlayer(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => SessionVideoPlayerScreen(
          session: session,
          sessionRepository: sessionRepository,
        ),
      ),
    );
  }

  Future<void> _deleteFromDetail(BuildContext context) async {
    final confirmed = await _confirmDeleteDialog(context, [session]);
    if (!confirmed || !context.mounted) {
      return;
    }
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    await sessionRepository.deleteSession(session);
    await onDeleted();
    if (!context.mounted) {
      return;
    }
    navigator.pop();
    messenger.showSnackBar(const SnackBar(content: Text('Deleted session.')));
  }
}

class _SessionThumbnail extends StatelessWidget {
  const _SessionThumbnail({
    required this.session,
    required this.sessionRepository,
  });

  final SessionSummary session;
  final SessionRepository sessionRepository;

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<TrackPoint>>(
      future: sessionRepository.loadTrackPoints(session),
      builder: (context, snapshot) {
        final points = snapshot.data ?? const <TrackPoint>[];
        if (points.isEmpty) {
          return DecoratedBox(
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Center(child: Icon(Icons.route, size: 34)),
          );
        }
        return _SessionMap(points: points, compact: true);
      },
    );
  }
}

class _SessionMap extends StatelessWidget {
  const _SessionMap({required this.points, this.compact = false});

  final List<TrackPoint> points;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final latLngs = points
        .map((point) => LatLng(point.lat, point.lon))
        .toList();
    final center = latLngs.isEmpty ? const LatLng(0, 0) : latLngs.first;
    final bounds = latLngs.length > 1 ? LatLngBounds.fromPoints(latLngs) : null;
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: FlutterMap(
        options: MapOptions(
          initialCenter: center,
          initialZoom: latLngs.isEmpty ? 2 : 15,
          initialCameraFit: bounds == null
              ? null
              : CameraFit.bounds(
                  bounds: bounds,
                  padding: EdgeInsets.all(compact ? 18 : 32),
                  maxZoom: 17,
                ),
          interactionOptions: compact
              ? const InteractionOptions(flags: InteractiveFlag.none)
              : const InteractionOptions(),
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
                  strokeWidth: compact ? 3 : 5,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ],
            ),
          if (latLngs.isNotEmpty)
            MarkerLayer(
              markers: [
                Marker(
                  point: latLngs.first,
                  width: compact ? 20 : 28,
                  height: compact ? 20 : 28,
                  child: const Icon(Icons.trip_origin, color: Colors.green),
                ),
                Marker(
                  point: latLngs.last,
                  width: compact ? 20 : 28,
                  height: compact ? 20 : 28,
                  child: const Icon(Icons.flag, color: Colors.redAccent),
                ),
              ],
            ),
        ],
      ),
    );
  }
}

class _SessionFact extends StatelessWidget {
  const _SessionFact({required this.icon, required this.value});

  final IconData icon;
  final String value;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 18),
            const SizedBox(width: 6),
            Text(
              value,
              style: Theme.of(
                context,
              ).textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w800),
            ),
          ],
        ),
      ),
    );
  }
}

class _SummaryTile extends StatelessWidget {
  const _SummaryTile({
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
      height: 86,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 22),
              const Spacer(),
              Text(
                label,
                style: theme.textTheme.labelMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              Text(
                value,
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w900,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  const _DetailRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 108,
            child: Text(label, style: Theme.of(context).textTheme.labelLarge),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }
}

class _DeleteBackground extends StatelessWidget {
  const _DeleteBackground();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.error,
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Align(
        alignment: Alignment.centerRight,
        child: Padding(
          padding: EdgeInsets.only(right: 24),
          child: Icon(Icons.delete_forever, color: Colors.white, size: 34),
        ),
      ),
    );
  }
}

Future<bool> _confirmDeleteDialog(
  BuildContext context,
  List<SessionSummary> sessions,
) async {
  final count = sessions.length;
  final confirmed = await showDialog<bool>(
    context: context,
    builder: (context) {
      return AlertDialog(
        title: Text(count == 1 ? 'Delete session?' : 'Delete $count sessions?'),
        content: Text(
          count == 1
              ? 'This removes the video segment, sidecar, GPX, thumbnails, and session folder from disk.'
              : 'This removes all selected video segments, sidecars, GPX files, thumbnails, and session folders from disk.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.error,
              foregroundColor: Theme.of(context).colorScheme.onError,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      );
    },
  );
  return confirmed ?? false;
}

String _deleteSnackText(int count) {
  if (count == 1) {
    return 'Deleted session.';
  }
  return 'Deleted $count sessions.';
}

double _trackDistance(List<TrackPoint> points) {
  if (points.length < 2) {
    return 0;
  }
  const distance = Distance();
  var meters = 0.0;
  for (var index = 1; index < points.length; index += 1) {
    meters += distance.as(
      LengthUnit.Meter,
      LatLng(points[index - 1].lat, points[index - 1].lon),
      LatLng(points[index].lat, points[index].lon),
    );
  }
  return meters;
}

String _formatDistance(double meters) {
  if (meters >= 1000) {
    return '${(meters / 1000).toStringAsFixed(1)} km';
  }
  return '${meters.toStringAsFixed(0)} m';
}

String _formatDuration(Duration duration) {
  final hours = duration.inHours;
  final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
  final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
  return '$hours:$minutes:$seconds';
}

String _formatBytes(int bytes) {
  const gb = 1024 * 1024 * 1024;
  const mb = 1024 * 1024;
  if (bytes >= gb) {
    return '${(bytes / gb).toStringAsFixed(1)} GB';
  }
  return '${(bytes / mb).toStringAsFixed(1)} MB';
}
