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
      final externalAvailable = sessions.any((session) => session.isExternal)
          ? await storageService.externalAvailable()
          : false;
      final markedSessions = sessions
          .map(
            (session) => session.isExternal
                ? session.copyWith(storageAvailable: externalAvailable)
                : session.copyWith(storageAvailable: true),
          )
          .toList();
      markedSessions.sort((a, b) => b.startedAtUtc.compareTo(a.startedAtUtc));
      return markedSessions;
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
    await _writeIndex(sessions);
  }

  Future<void> deleteSession(SessionSummary summary) async {
    final sessions = await listSessions();
    sessions.removeWhere((session) => session.id == summary.id);
    await _writeIndex(sessions);
    await _deleteSessionArtifacts(summary);
  }

  Future<void> deleteSessions(Iterable<SessionSummary> summaries) async {
    final ids = summaries.map((summary) => summary.id).toSet();
    final sessions = await listSessions();
    sessions.removeWhere((session) => ids.contains(session.id));
    await _writeIndex(sessions);
    for (final summary in summaries) {
      await _deleteSessionArtifacts(summary);
    }
  }

  Future<List<TrackPoint>> loadTrackPoints(SessionSummary summary) async {
    final raw = await _readSessionText(summary, summary.gpxPath);
    if (raw == null) {
      return _fallbackPoints(summary);
    }
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

  Future<void> _writeIndex(List<SessionSummary> sessions) async {
    final file = await _indexFile();
    await file.writeAsString(
      const JsonEncoder.withIndent(
        '  ',
      ).convert(sessions.map((session) => session.toJson()).toList()),
    );
  }

  Future<void> _deleteSessionArtifacts(SessionSummary summary) async {
    if (summary.isExternal) {
      await _deleteExternalSessionArtifacts(summary);
      return;
    }

    final root = await storageService.recordingsRoot();
    final directory = Directory(summary.directoryPath);
    if (_isInsideRoot(root, directory) && await directory.exists()) {
      await directory.delete(recursive: true);
      return;
    }

    final paths = <String>{
      summary.videoPath,
      summary.sidecarPath,
      summary.gpxPath,
    };
    for (final path in paths.where((path) => path.isNotEmpty)) {
      final file = File(path);
      if (_isInsideRoot(root, file) && await file.exists()) {
        await file.delete();
      }
    }
  }

  Future<String?> _readSessionText(SessionSummary summary, String path) async {
    if (summary.isExternal) {
      if (!await storageService.externalAvailable()) {
        return null;
      }
      return storageService.readExternalText(path);
    }

    final file = File(path);
    try {
      if (!await file.exists()) {
        return null;
      }
      return file.readAsString();
    } on FileSystemException {
      return null;
    }
  }

  Future<void> _deleteExternalSessionArtifacts(SessionSummary summary) async {
    if (!await storageService.externalAvailable()) {
      return;
    }
    final paths = <String>{
      summary.videoPath,
      summary.sidecarPath,
      summary.gpxPath,
    };
    for (final path in paths.where((path) => path.isNotEmpty)) {
      await storageService.deleteExternalFile(path);
    }
  }

  bool _isInsideRoot(Directory root, FileSystemEntity entity) {
    final rootPath = '${root.absolute.path}${Platform.pathSeparator}';
    final path = entity.absolute.path;
    return path == root.absolute.path || path.startsWith(rootPath);
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
