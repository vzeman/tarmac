import 'dart:convert';
import 'dart:io';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import 'storage_service.dart';

class SessionRepository {
  SessionRepository({StorageService? storageService})
    : storageService = storageService ?? StorageService();

  final StorageService storageService;

  Future<Directory> createSessionDirectory(String sessionId) {
    return storageService.createSessionDirectory(sessionId);
  }

  Future<List<SessionSummary>> listSessions() async {
    final file = await _indexFile();
    if (!await file.exists()) {
      return [];
    }
    try {
      final raw = await file.readAsString();
      final decoded = jsonDecode(raw) as List<dynamic>;
      final sessions = decoded
          .whereType<Map<String, dynamic>>()
          .map(SessionSummary.fromJson)
          .where((session) => session.id.isNotEmpty)
          .toList();
      sessions.sort((a, b) => b.startedAtUtc.compareTo(a.startedAtUtc));
      return sessions;
    } on FormatException {
      return [];
    } on TypeError {
      return [];
    }
  }

  Future<void> saveSummary(SessionSummary summary) async {
    final sessions = await listSessions();
    sessions.removeWhere((session) => session.id == summary.id);
    sessions.insert(0, summary);
    final file = await _indexFile();
    await file.writeAsString(
      const JsonEncoder.withIndent(
        '  ',
      ).convert(sessions.map((session) => session.toJson()).toList()),
    );
  }

  Future<List<TrackPoint>> loadTrackPoints(SessionSummary summary) async {
    final file = File(summary.gpxPath);
    if (!await file.exists()) {
      return _fallbackPoints(summary);
    }
    final raw = await file.readAsString();
    final matches = RegExp(
      r'<trkpt lat="([^"]+)" lon="([^"]+)">.*?<time>([^<]+)</time>',
      dotAll: true,
    ).allMatches(raw);
    final points = <TrackPoint>[];
    for (final match in matches) {
      final lat = double.tryParse(match.group(1) ?? '');
      final lon = double.tryParse(match.group(2) ?? '');
      final time = DateTime.tryParse(match.group(3) ?? '')?.toUtc();
      if (lat != null && lon != null) {
        points.add(
          TrackPoint(lat: lat, lon: lon, utcMs: time?.millisecondsSinceEpoch),
        );
      }
    }
    return points.isEmpty ? _fallbackPoints(summary) : points;
  }

  String createSessionId(DateTime utc) {
    final normalized = utc.toUtc();
    String two(int value) => value.toString().padLeft(2, '0');
    String three(int value) => value.toString().padLeft(3, '0');
    return 'rs_${normalized.year}'
        '${two(normalized.month)}'
        '${two(normalized.day)}_'
        '${two(normalized.hour)}'
        '${two(normalized.minute)}'
        '${two(normalized.second)}'
        '${three(normalized.millisecond)}Z';
  }

  Future<File> _indexFile() async {
    final root = await storageService.recordingsRoot();
    return File('${root.path}/sessions_index.json');
  }

  List<TrackPoint> _fallbackPoints(SessionSummary summary) {
    final points = <TrackPoint>[];
    if (summary.startLat != null && summary.startLon != null) {
      points.add(TrackPoint(lat: summary.startLat!, lon: summary.startLon!));
    }
    if (summary.endLat != null && summary.endLon != null) {
      points.add(TrackPoint(lat: summary.endLat!, lon: summary.endLon!));
    }
    return points;
  }
}
