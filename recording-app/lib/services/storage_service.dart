import 'dart:async';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../models/session_summary.dart';
import '../settings/app_settings.dart';

enum StorageTargetType { internal, external }

class StorageTarget {
  const StorageTarget._({
    required this.type,
    required this.freeBytes,
    this.externalRequested = false,
    this.externalUnavailable = false,
  });

  factory StorageTarget.internal({
    int? freeBytes,
    bool externalRequested = false,
    bool externalUnavailable = false,
  }) {
    return StorageTarget._(
      type: StorageTargetType.internal,
      freeBytes: freeBytes,
      externalRequested: externalRequested,
      externalUnavailable: externalUnavailable,
    );
  }

  factory StorageTarget.external({int? freeBytes}) {
    return StorageTarget._(
      type: StorageTargetType.external,
      freeBytes: freeBytes,
    );
  }

  final StorageTargetType type;
  final int? freeBytes;
  final bool externalRequested;
  final bool externalUnavailable;

  bool get isExternal => type == StorageTargetType.external;

  String get label => isExternal ? 'External' : 'Internal';
}

class ExternalFileAccess {
  ExternalFileAccess._(this._storageService, this.path);

  final StorageService _storageService;
  final String path;
  bool _released = false;

  void release() {
    if (_released) {
      return;
    }
    _released = true;
    unawaited(_storageService.stopExternalFileAccess(path));
  }
}

class StorageService {
  static const MethodChannel _storageChannel = MethodChannel(
    'roadsurvey_recorder/storage',
  );

  Future<Directory> recordingsRoot() async {
    final base = await getApplicationDocumentsDirectory();
    final root = Directory('${base.path}/RoadSurveyRecorder');
    if (!await root.exists()) {
      await root.create(recursive: true);
    }
    return root;
  }

  Future<Directory> createSessionDirectory(String sessionId) async {
    final root = await recordingsRoot();
    final directory = Directory('${root.path}/$sessionId');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
    return directory;
  }

  Future<String> resolveInternalPath(String storedPath) async {
    final path = _normalizeStoredPath(storedPath);
    if (!_isResolvableInternalPath(path)) {
      return path;
    }

    final documentsPath = await _documentsDirectoryPath();
    if (!_isAbsolutePath(path)) {
      return _joinDocumentsPath(documentsPath, path);
    }

    if (await _entityExists(path)) {
      return path;
    }

    final staleRelativePath = _relativePathAfterDocuments(path);
    if (staleRelativePath != null) {
      return _joinDocumentsPath(documentsPath, staleRelativePath);
    }

    return path;
  }

  Future<SessionSummary> summaryWithPortableInternalPaths(
    SessionSummary summary,
  ) async {
    if (summary.isExternal) {
      return summary;
    }
    return summary.copyWith(
      directoryPath: await _portableInternalPath(summary.directoryPath),
      videoPath: await _portableInternalPath(summary.videoPath),
      sidecarPath: await _portableInternalPath(summary.sidecarPath),
      gpxPath: await _portableInternalPath(summary.gpxPath),
    );
  }

  Future<int?> freeBytesForPath(String path) async {
    return _invokeInt('freeBytes', {'path': path});
  }

  Future<int?> freeBytesForExternal() async {
    return _invokeInt('externalFreeBytes');
  }

  Future<int?> _invokeInt(String method, [Map<String, Object?>? args]) async {
    try {
      final value = await _storageChannel.invokeMethod<Object?>(method, args);
      if (value is int) {
        return value;
      }
      if (value is num) {
        return value.round();
      }
      return null;
    } on PlatformException {
      return null;
    } on MissingPluginException {
      return null;
    }
  }

  Future<int?> freeBytesForRecordingsRoot() async {
    final root = await recordingsRoot();
    return freeBytesForPath(root.path);
  }

  Future<bool> chooseExternal() async {
    try {
      return await _storageChannel.invokeMethod<bool>(
            'pickExternalDirectory',
          ) ??
          false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  Future<bool> externalAvailable() async {
    try {
      return await _storageChannel.invokeMethod<bool>('isExternalAvailable') ??
          false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  Future<StorageTarget> activeTarget(
    AppSettings settings, {
    bool forceInternal = false,
  }) async {
    if (!forceInternal &&
        settings.storageLocation == StorageLocation.external) {
      final available = await externalAvailable();
      if (available) {
        return StorageTarget.external(freeBytes: await freeBytesForExternal());
      }
      return StorageTarget.internal(
        freeBytes: await freeBytesForRecordingsRoot(),
        externalRequested: true,
        externalUnavailable: true,
      );
    }
    return StorageTarget.internal(
      freeBytes: await freeBytesForRecordingsRoot(),
    );
  }

  Future<SessionSummary> finalizeToTarget({
    required SessionSummary summary,
    required StorageTarget target,
  }) async {
    if (!target.isExternal) {
      return summary.copyWith(
        storageLocation: 'internal',
        storageAvailable: true,
      );
    }

    if (!await externalAvailable()) {
      throw StateError('External storage is not available.');
    }

    final directory = Directory(summary.directoryPath);
    final files = await _filesIn(directory);
    if (files.isEmpty) {
      throw StateError('No finalized session files were found.');
    }

    final movedPaths = <String, String>{};
    var totalBytes = 0;
    for (final file in files) {
      totalBytes += await file.length();
      final relativePath = '${summary.id}/${_relativePath(directory, file)}';
      movedPaths[file.path] = await _moveFileToExternal(
        srcPath: file.path,
        filename: relativePath,
      );
    }

    if (await directory.exists()) {
      await directory.delete(recursive: true);
    }

    final videoPath = movedPaths[summary.videoPath] ?? summary.videoPath;
    final sidecarPath = movedPaths[summary.sidecarPath] ?? summary.sidecarPath;
    final gpxPath = movedPaths[summary.gpxPath] ?? summary.gpxPath;

    return summary.copyWith(
      directoryPath: _externalDirectoryPath(summary.id, videoPath),
      videoPath: videoPath,
      sidecarPath: sidecarPath,
      gpxPath: gpxPath,
      totalBytes: totalBytes,
      storageLocation: 'external',
      storageAvailable: true,
    );
  }

  Future<String?> readExternalText(String path) async {
    try {
      return await _storageChannel.invokeMethod<String>('readExternalText', {
        'path': path,
      });
    } on PlatformException {
      return null;
    } on MissingPluginException {
      return null;
    }
  }

  Future<bool> deleteExternalFile(String path) async {
    try {
      return await _storageChannel.invokeMethod<bool>('deleteExternalFile', {
            'path': path,
          }) ??
          false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  Future<ExternalFileAccess?> startExternalFileAccess(String path) async {
    try {
      final available =
          await _storageChannel.invokeMethod<bool>('startExternalAccess', {
            'path': path,
          }) ??
          false;
      return available ? ExternalFileAccess._(this, path) : null;
    } on PlatformException {
      return null;
    } on MissingPluginException {
      return null;
    }
  }

  Future<void> stopExternalFileAccess(String path) async {
    try {
      await _storageChannel.invokeMethod<bool>('stopExternalAccess', {
        'path': path,
      });
    } on PlatformException {
      return;
    } on MissingPluginException {
      return;
    }
  }

  Future<int> directorySize(Directory directory) async {
    var total = 0;
    if (!await directory.exists()) {
      return total;
    }
    await for (final entity in directory.list(recursive: true)) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  Future<List<File>> _filesIn(Directory directory) async {
    if (!await directory.exists()) {
      return [];
    }
    final files = <File>[];
    await for (final entity in directory.list(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is File) {
        files.add(entity);
      }
    }
    files.sort((a, b) => a.path.compareTo(b.path));
    return files;
  }

  Future<String> _moveFileToExternal({
    required String srcPath,
    required String filename,
  }) async {
    final destination = await _storageChannel.invokeMethod<String>(
      'moveFileToExternal',
      {'srcPath': srcPath, 'filename': filename},
    );
    if (destination == null || destination.isEmpty) {
      throw StateError('External storage did not return a destination path.');
    }
    return destination;
  }

  String _relativePath(Directory root, File file) {
    final separator = Platform.pathSeparator;
    final rootPath = root.absolute.path;
    final prefix = rootPath.endsWith(separator)
        ? rootPath
        : '$rootPath$separator';
    final filePath = file.absolute.path;
    final raw = filePath.startsWith(prefix)
        ? filePath.substring(prefix.length)
        : _basename(filePath);
    return raw.replaceAll(separator, '/').replaceAll('\\', '/');
  }

  String _basename(String path) {
    final normalized = path.replaceAll('\\', '/');
    final slash = normalized.lastIndexOf('/');
    return slash >= 0 ? normalized.substring(slash + 1) : normalized;
  }

  String _externalDirectoryPath(String sessionId, String videoPath) {
    if (videoPath.startsWith('content://')) {
      return 'external://$sessionId';
    }
    final slash = videoPath.lastIndexOf('/');
    if (slash <= 0) {
      return 'external://$sessionId';
    }
    return videoPath.substring(0, slash);
  }

  Future<String> _documentsDirectoryPath() async {
    return (await getApplicationDocumentsDirectory()).path;
  }

  Future<String> _portableInternalPath(String storedPath) async {
    final path = _normalizeStoredPath(storedPath);
    if (!_isResolvableInternalPath(path)) {
      return path;
    }
    if (!_isAbsolutePath(path)) {
      return _normalizeRelativePath(path);
    }

    final documentsPath = await _documentsDirectoryPath();
    final currentRelativePath = _relativePathInside(documentsPath, path);
    if (currentRelativePath != null) {
      return currentRelativePath;
    }

    if (await _entityExists(path)) {
      return path;
    }

    final staleRelativePath = _relativePathAfterDocuments(path);
    if (staleRelativePath == null) {
      return path;
    }

    final migratedPath = _joinDocumentsPath(documentsPath, staleRelativePath);
    return await _entityExists(migratedPath) ? staleRelativePath : path;
  }

  String _normalizeStoredPath(String path) {
    final trimmed = path.trim();
    final uri = Uri.tryParse(trimmed);
    if (uri != null && uri.scheme == 'file') {
      return uri.toFilePath();
    }
    return trimmed;
  }

  bool _isResolvableInternalPath(String path) {
    if (path.isEmpty) {
      return false;
    }
    final uri = Uri.tryParse(path);
    return uri == null || !uri.hasScheme || uri.scheme == 'file';
  }

  bool _isAbsolutePath(String path) {
    return path.startsWith('/') || RegExp(r'^[A-Za-z]:[\\/]').hasMatch(path);
  }

  String _joinDocumentsPath(String documentsPath, String relativePath) {
    final root = _stripTrailingSeparators(documentsPath);
    final localRelativePath = _normalizeRelativePath(
      relativePath,
    ).replaceAll('/', Platform.pathSeparator);
    return '$root${Platform.pathSeparator}$localRelativePath';
  }

  String _normalizeRelativePath(String path) {
    var normalized = path.replaceAll('\\', '/');
    while (normalized.startsWith('/')) {
      normalized = normalized.substring(1);
    }
    return normalized;
  }

  String? _relativePathInside(String rootPath, String path) {
    final root = _stripTrailingSlashes(rootPath.replaceAll('\\', '/'));
    final normalizedPath = path.replaceAll('\\', '/');
    if (normalizedPath == root) {
      return '';
    }
    final prefix = '$root/';
    if (!normalizedPath.startsWith(prefix)) {
      return null;
    }
    return _normalizeRelativePath(normalizedPath.substring(prefix.length));
  }

  String? _relativePathAfterDocuments(String path) {
    final normalizedPath = path.replaceAll('\\', '/');
    const marker = '/Documents/';
    final markerIndex = normalizedPath.indexOf(marker);
    if (markerIndex < 0) {
      return null;
    }
    final relativePath = normalizedPath.substring(markerIndex + marker.length);
    return relativePath.isEmpty ? null : _normalizeRelativePath(relativePath);
  }

  String _stripTrailingSeparators(String path) {
    final separator = Platform.pathSeparator;
    var stripped = path;
    while (stripped.length > 1 && stripped.endsWith(separator)) {
      stripped = stripped.substring(0, stripped.length - separator.length);
    }
    return stripped;
  }

  String _stripTrailingSlashes(String path) {
    var stripped = path;
    while (stripped.length > 1 && stripped.endsWith('/')) {
      stripped = stripped.substring(0, stripped.length - 1);
    }
    return stripped;
  }

  Future<bool> _entityExists(String path) async {
    try {
      return await FileSystemEntity.type(path) != FileSystemEntityType.notFound;
    } on FileSystemException {
      return false;
    }
  }
}
