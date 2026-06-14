import 'dart:convert';
import 'dart:io';

import '../models/session_summary.dart';
import '../models/telemetry.dart';
import 'storage_service.dart';

class SessionSharePackage {
  SessionSharePackage({
    required this.availableFiles,
    required this.unavailableFiles,
    List<ExternalFileAccess> externalAccesses = const [],
  }) : _externalAccesses = externalAccesses;

  final List<SessionShareFile> availableFiles;
  final List<UnavailableSessionShareFile> unavailableFiles;
  final List<ExternalFileAccess> _externalAccesses;
  bool _released = false;

  void release() {
    if (_released) {
      return;
    }
    _released = true;
    for (final access in _externalAccesses) {
      access.release();
    }
  }
}

class SessionShareFile {
  const SessionShareFile({
    required this.path,
    required this.displayName,
    required this.mimeType,
  });

  final String path;
  final String displayName;
  final String mimeType;
}

class UnavailableSessionShareFile {
  const UnavailableSessionShareFile({
    required this.path,
    required this.displayName,
    required this.reason,
  });

  final String path;
  final String displayName;
  final String reason;
}

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

  Future<SessionSharePackage> resolveShareableFiles(
    SessionSummary summary,
  ) async {
    final externalAccesses = <ExternalFileAccess>[];
    try {
      final candidates = await _shareCandidates(summary, externalAccesses);
      final available = <SessionShareFile>[];
      final unavailable = <UnavailableSessionShareFile>[];
      final seenPaths = <String>{};

      for (final candidate in candidates) {
        final path = _normalizeSharePath(candidate.path);
        if (path == null || !seenPaths.add(path)) {
          continue;
        }
        final displayName = _displayName(path, candidate.kind);
        final mimeType = _mimeTypeForKind(candidate.kind);
        if (_isContentUri(path)) {
          final access = summary.isExternal
              ? await storageService.startExternalFileAccess(path)
              : null;
          access?.release();
          unavailable.add(
            UnavailableSessionShareFile(
              path: path,
              displayName: displayName,
              reason: access == null
                  ? 'external storage unavailable'
                  : 'external Android URI',
            ),
          );
          continue;
        }

        ExternalFileAccess? fileAccess;
        if (summary.isExternal) {
          fileAccess = await storageService.startExternalFileAccess(path);
          if (fileAccess == null) {
            unavailable.add(
              UnavailableSessionShareFile(
                path: path,
                displayName: displayName,
                reason: 'external storage unavailable',
              ),
            );
            continue;
          }
        }

        final exists = await _fileExists(path);
        if (!exists) {
          fileAccess?.release();
          unavailable.add(
            UnavailableSessionShareFile(
              path: path,
              displayName: displayName,
              reason: 'missing',
            ),
          );
          continue;
        }

        if (fileAccess != null) {
          externalAccesses.add(fileAccess);
        }
        available.add(
          SessionShareFile(
            path: path,
            displayName: displayName,
            mimeType: mimeType,
          ),
        );
      }

      return SessionSharePackage(
        availableFiles: available,
        unavailableFiles: unavailable,
        externalAccesses: externalAccesses,
      );
    } on Object {
      for (final access in externalAccesses) {
        access.release();
      }
      rethrow;
    }
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

  Future<List<_SessionShareCandidate>> _shareCandidates(
    SessionSummary summary,
    List<ExternalFileAccess> externalAccesses,
  ) async {
    final candidates = <_SessionShareCandidate>[
      if (summary.videoPath.trim().isNotEmpty)
        _SessionShareCandidate(summary.videoPath, _SessionShareKind.video),
      if (summary.sidecarPath.trim().isNotEmpty)
        _SessionShareCandidate(summary.sidecarPath, _SessionShareKind.sidecar),
      if (summary.gpxPath.trim().isNotEmpty)
        _SessionShareCandidate(summary.gpxPath, _SessionShareKind.gpx),
    ];

    final directoryPath = _normalizeSharePath(summary.directoryPath);
    if (directoryPath == null || _isContentUri(directoryPath)) {
      return _sortShareCandidates(candidates);
    }

    ExternalFileAccess? directoryAccess;
    if (summary.isExternal) {
      directoryAccess = await storageService.startExternalFileAccess(
        directoryPath,
      );
      if (directoryAccess == null) {
        return _sortShareCandidates(candidates);
      }
      externalAccesses.add(directoryAccess);
    }

    final directory = Directory(directoryPath);
    if (!await _directoryExists(directory)) {
      return _sortShareCandidates(candidates);
    }

    await for (final entity in directory.list(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is! File) {
        continue;
      }
      final kind = _shareKindForPath(entity.path);
      if (kind == null) {
        continue;
      }
      candidates.add(_SessionShareCandidate(entity.path, kind));
    }

    return _sortShareCandidates(candidates);
  }

  List<_SessionShareCandidate> _sortShareCandidates(
    List<_SessionShareCandidate> candidates,
  ) {
    return candidates.toList()..sort((a, b) {
      final kindComparison = a.kind.index.compareTo(b.kind.index);
      if (kindComparison != 0) {
        return kindComparison;
      }
      return _filenameFromPath(a.path).compareTo(_filenameFromPath(b.path));
    });
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

enum _SessionShareKind { video, sidecar, gpx }

class _SessionShareCandidate {
  const _SessionShareCandidate(this.path, this.kind);

  final String path;
  final _SessionShareKind kind;
}

_SessionShareKind? _shareKindForPath(String path) {
  final name = _filenameFromPath(path).toLowerCase();
  if (name.endsWith('.mp4')) {
    return _SessionShareKind.video;
  }
  if (name.endsWith('.track.json')) {
    return _SessionShareKind.sidecar;
  }
  if (name.endsWith('.gpx')) {
    return _SessionShareKind.gpx;
  }
  return null;
}

String _mimeTypeForKind(_SessionShareKind kind) {
  return switch (kind) {
    _SessionShareKind.video => 'video/mp4',
    _SessionShareKind.sidecar => 'application/json',
    _SessionShareKind.gpx => 'application/gpx+xml',
  };
}

String _displayName(String path, _SessionShareKind kind) {
  final name = _filenameFromPath(path);
  if (name.isNotEmpty && name.contains('.')) {
    return name;
  }
  return switch (kind) {
    _SessionShareKind.video => 'video.mp4',
    _SessionShareKind.sidecar => 'track.json',
    _SessionShareKind.gpx => 'track.gpx',
  };
}

String _filenameFromPath(String path) {
  final uri = Uri.tryParse(path);
  if (uri != null && uri.hasScheme && uri.scheme != 'file') {
    final segment = uri.pathSegments.isEmpty ? path : uri.pathSegments.last;
    return _basename(Uri.decodeComponent(segment).split(':').last);
  }
  return _basename(path);
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final slash = normalized.lastIndexOf('/');
  return slash >= 0 ? normalized.substring(slash + 1) : normalized;
}

String? _normalizeSharePath(String rawPath) {
  final trimmed = rawPath.trim();
  if (trimmed.isEmpty) {
    return null;
  }
  final uri = Uri.tryParse(trimmed);
  if (uri != null && uri.scheme == 'file') {
    return uri.toFilePath();
  }
  return trimmed;
}

bool _isContentUri(String path) {
  return Uri.tryParse(path)?.scheme == 'content';
}

Future<bool> _fileExists(String path) async {
  try {
    return await File(path).exists();
  } on FileSystemException {
    return false;
  }
}

Future<bool> _directoryExists(Directory directory) async {
  try {
    return await directory.exists();
  } on FileSystemException {
    return false;
  }
}
