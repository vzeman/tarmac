import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

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

  Future<int?> freeBytesForPath(String path) async {
    try {
      final value = await _storageChannel.invokeMethod<int>('freeBytes', {
        'path': path,
      });
      return value;
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
}
