class RecordingClock {
  RecordingClock(this.startUtc);

  final DateTime startUtc;
  final Stopwatch _stopwatch = Stopwatch();

  void start() {
    if (!_stopwatch.isRunning) {
      _stopwatch.start();
    }
  }

  void stop() {
    _stopwatch.stop();
  }

  int get ptsMs => _stopwatch.elapsedMilliseconds;

  int get utcMs => startUtc.millisecondsSinceEpoch + ptsMs;

  int get elapsedMs => _stopwatch.elapsedMilliseconds;

  DateTime get nowUtc {
    return DateTime.fromMillisecondsSinceEpoch(utcMs, isUtc: true);
  }
}
