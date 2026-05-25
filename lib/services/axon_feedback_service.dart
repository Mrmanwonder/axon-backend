import 'package:audioplayers/audioplayers.dart';

class AxonFeedbackService {
  AxonFeedbackService._();

  static final AudioPlayer _goalPlayer = AudioPlayer()
    ..setReleaseMode(ReleaseMode.stop)
    ..setPlayerMode(PlayerMode.lowLatency);
  static final AudioPlayer _aliveLoopPlayer = AudioPlayer()
    ..setReleaseMode(ReleaseMode.loop)
    ..setPlayerMode(PlayerMode.lowLatency)
    ..setVolume(0.14);

  static Future<void> playGoalUnlockChime() async {
    try {
      await _goalPlayer.stop();
      await _goalPlayer.play(AssetSource('audio/unlock_chime.wav'));
    } catch (_) {
      // Audio feedback is non-critical.
    }
  }

  static Future<void> playMilestoneChime() => playGoalUnlockChime();

  static Future<void> startActiveLoop() async {
    try {
      final currentState = _aliveLoopPlayer.state;
      if (currentState == PlayerState.playing) {
        return;
      }
      await _aliveLoopPlayer.stop();
      await _aliveLoopPlayer.play(AssetSource('audio/session_alive_loop.wav'));
    } catch (_) {
      // Audio feedback is non-critical.
    }
  }

  static Future<void> stopActiveLoop() async {
    try {
      await _aliveLoopPlayer.stop();
    } catch (_) {
      // Audio feedback is non-critical.
    }
  }
}
