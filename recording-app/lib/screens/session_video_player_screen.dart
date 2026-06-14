import 'dart:io';

import 'package:chewie/chewie.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../models/session_summary.dart';
import '../services/session_repository.dart';
import '../services/storage_service.dart';

class SessionVideoPlayerScreen extends StatefulWidget {
  const SessionVideoPlayerScreen({
    super.key,
    required this.session,
    required this.sessionRepository,
  });

  final SessionSummary session;
  final SessionRepository sessionRepository;

  @override
  State<SessionVideoPlayerScreen> createState() =>
      _SessionVideoPlayerScreenState();
}

class _SessionVideoPlayerScreenState extends State<SessionVideoPlayerScreen> {
  VideoPlayerController? _videoController;
  ChewieController? _chewieController;
  ExternalFileAccess? _externalAccess;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  @override
  void dispose() {
    _chewieController?.dispose();
    _videoController?.dispose();
    _externalAccess?.release();
    super.dispose();
  }

  Future<void> _initialize() async {
    final storedPath = widget.session.videoPath.trim();
    if (storedPath.isEmpty) {
      _showError('This session does not have a video segment.');
      return;
    }

    final path = await widget.sessionRepository.resolveSessionArtifactPath(
      widget.session,
      storedPath,
    );
    final isContentUri = _isContentUri(path);
    if (widget.session.isExternal) {
      final access = await widget.sessionRepository.storageService
          .startExternalFileAccess(path);
      if (!mounted) {
        access?.release();
        return;
      }
      if (access == null) {
        _showError(
          'The video is not available. Connect the external storage and try again.',
        );
        return;
      }
      _externalAccess = access;
    }

    if (isContentUri && defaultTargetPlatform != TargetPlatform.android) {
      _showError('This external video URI can only be played on Android.');
      return;
    }

    if (!isContentUri && !await _fileExists(path)) {
      _showError('Video file not found at $path');
      return;
    }

    final controller = isContentUri
        ? VideoPlayerController.contentUri(Uri.parse(path))
        : VideoPlayerController.file(File(path));
    _videoController = controller;

    try {
      await controller.initialize();
      if (!mounted) {
        return;
      }
      final aspectRatio = controller.value.aspectRatio > 0
          ? controller.value.aspectRatio
          : 16 / 9;
      _chewieController = ChewieController(
        videoPlayerController: controller,
        aspectRatio: aspectRatio,
        autoPlay: false,
        looping: false,
        allowFullScreen: true,
        allowMuting: true,
        allowPlaybackSpeedChanging: true,
        showControls: true,
        errorBuilder: (context, message) {
          return _VideoUnavailable(message: message);
        },
      );
      setState(() {
        _loading = false;
      });
    } on Object catch (error) {
      await controller.dispose();
      if (!mounted) {
        return;
      }
      _videoController = null;
      _showError('Video decode/init failed: $error');
    }
  }

  Future<bool> _fileExists(String path) async {
    try {
      return await File(path).exists();
    } on FileSystemException {
      return false;
    }
  }

  void _showError(String message) {
    _externalAccess?.release();
    _externalAccess = null;
    if (!mounted) {
      return;
    }
    setState(() {
      _loading = false;
      _error = message;
    });
  }

  bool _isContentUri(String path) {
    return Uri.tryParse(path)?.scheme == 'content';
  }

  @override
  Widget build(BuildContext context) {
    final title = _segmentName(widget.session.videoPath);
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: SafeArea(
        top: false,
        child: ColoredBox(
          color: Colors.black,
          child: Center(child: _buildBody(context)),
        ),
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return const CircularProgressIndicator();
    }

    final error = _error;
    final chewieController = _chewieController;
    if (error != null || chewieController == null) {
      return _VideoUnavailable(
        message: error ?? 'The video file could not be opened.',
      );
    }

    return Chewie(controller: chewieController);
  }
}

class _VideoUnavailable extends StatelessWidget {
  const _VideoUnavailable({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.videocam_off_outlined,
            color: theme.colorScheme.onInverseSurface,
            size: 42,
          ),
          const SizedBox(height: 12),
          Text(
            message,
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyLarge?.copyWith(
              color: theme.colorScheme.onInverseSurface,
            ),
          ),
        ],
      ),
    );
  }
}

String _segmentName(String path) {
  if (path.isEmpty) {
    return 'Video';
  }
  final normalized = path.replaceAll('\\', '/');
  final slash = normalized.lastIndexOf('/');
  if (slash >= 0 && slash < normalized.length - 1) {
    return normalized.substring(slash + 1);
  }
  return normalized;
}
