import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import '../services/session_repository.dart';

class SessionsScreen extends StatelessWidget {
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
  Widget build(BuildContext context) {
    if (sessions.isEmpty) {
      return RefreshIndicator(
        onRefresh: onRefresh,
        child: ListView(
          children: const [
            SizedBox(height: 160),
            Center(child: Text('No recorded sessions')),
          ],
        ),
      );
    }
    return RefreshIndicator(
      onRefresh: onRefresh,
      child: ListView.separated(
        padding: const EdgeInsets.all(12),
        itemBuilder: (context, index) {
          final session = sessions[index];
          return ListTile(
            leading: const Icon(Icons.route),
            title: Text(
              DateFormat.yMMMd().add_Hm().format(
                session.startedAtUtc.toLocal(),
              ),
            ),
            subtitle: Text(
              '${_formatDuration(Duration(milliseconds: session.durationMs))}  '
              '${session.frameCount} frames  '
              '${session.gpsSampleCount} GPS  '
              '${session.imuSampleCount} IMU',
            ),
            trailing: Text(_formatBytes(session.totalBytes)),
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute<void>(
                  builder: (_) => SessionDetailScreen(
                    session: session,
                    sessionRepository: sessionRepository,
                  ),
                ),
              );
            },
          );
        },
        separatorBuilder: (context, index) => const Divider(height: 1),
        itemCount: sessions.length,
      ),
    );
  }
}

class SessionDetailScreen extends StatelessWidget {
  const SessionDetailScreen({
    super.key,
    required this.session,
    required this.sessionRepository,
  });

  final SessionSummary session;
  final SessionRepository sessionRepository;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(session.id)),
      body: FutureBuilder<List<TrackPoint>>(
        future: sessionRepository.loadTrackPoints(session),
        builder: (context, snapshot) {
          final points = snapshot.data ?? [];
          return ListView(
            padding: const EdgeInsets.all(12),
            children: [
              SizedBox(height: 300, child: _SessionMap(points: points)),
              const SizedBox(height: 12),
              _DetailRow(
                label: 'Started',
                value: session.startedAtUtc.toLocal().toString(),
              ),
              _DetailRow(
                label: 'Duration',
                value: _formatDuration(
                  Duration(milliseconds: session.durationMs),
                ),
              ),
              _DetailRow(label: 'Frames', value: session.frameCount.toString()),
              _DetailRow(
                label: 'GPS samples',
                value: session.gpsSampleCount.toString(),
              ),
              _DetailRow(
                label: 'IMU samples',
                value: session.imuSampleCount.toString(),
              ),
              _DetailRow(
                label: 'Size',
                value: _formatBytes(session.totalBytes),
              ),
              _DetailRow(label: 'Video', value: session.videoPath),
              _DetailRow(label: 'Sidecar', value: session.sidecarPath),
              _DetailRow(label: 'GPX', value: session.gpxPath),
            ],
          );
        },
      ),
    );
  }
}

class _SessionMap extends StatelessWidget {
  const _SessionMap({required this.points});

  final List<TrackPoint> points;

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
                  padding: const EdgeInsets.all(32),
                  maxZoom: 17,
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
                  point: latLngs.first,
                  width: 28,
                  height: 28,
                  child: const Icon(Icons.trip_origin, color: Colors.green),
                ),
                Marker(
                  point: latLngs.last,
                  width: 28,
                  height: 28,
                  child: const Icon(Icons.flag, color: Colors.redAccent),
                ),
              ],
            ),
        ],
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
            width: 96,
            child: Text(label, style: Theme.of(context).textTheme.labelLarge),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }
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
