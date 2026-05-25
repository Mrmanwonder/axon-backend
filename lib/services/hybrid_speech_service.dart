// lib/services/hybrid_speech_service.dart
//
// Hybrid Speech-to-Text engine:
//   Primary  → Device-native STT (instant partials + audio levels)
//   Fallback → Deepgram streaming (when native unavailable or low confidence)

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:permission_handler/permission_handler.dart';

import 'deepgram_service.dart';

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

enum SpeechEngine { native, deepgram }

class HybridTranscript {
  final String text;
  final bool isFinal;
  final double confidence;
  final SpeechEngine source;

  const HybridTranscript({
    required this.text,
    this.isFinal = false,
    this.confidence = 1.0,
    this.source = SpeechEngine.native,
  });
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

class HybridSpeechService {
  HybridSpeechService._();
  static final HybridSpeechService instance = HybridSpeechService._();

  // ── Engines ──────────────────────────────────────────────────────────────
  final SpeechToText _nativeStt = SpeechToText();
  bool _nativeInitialized = false;
  bool _deepgramAvailable = false;

  // ── State ────────────────────────────────────────────────────────────────
  SpeechEngine _activeEngine = SpeechEngine.native;
  bool _isListening = false;
  String _committedText = '';

  // Confidence tracking — switch to Deepgram after repeated low scores
  final List<double> _recentConfidences = [];
  static const double _confidenceThreshold = 0.65;
  static const int _lowConfidenceSwitchCount = 3;

  // ── Streams ──────────────────────────────────────────────────────────────
  StreamController<HybridTranscript>? _transcriptCtrl;
  StreamController<double>? _audioLevelCtrl;

  Stream<HybridTranscript> get transcriptStream =>
      (_transcriptCtrl ??= StreamController<HybridTranscript>.broadcast())
          .stream;

  Stream<double> get audioLevelStream =>
      (_audioLevelCtrl ??= StreamController<double>.broadcast()).stream;

  bool get isListening => _isListening;
  SpeechEngine get activeEngine => _activeEngine;

  // ── Deepgram subscription ───────────────────────────────────────────────
  StreamSubscription<DeepgramTranscript>? _deepgramSub;

  // ── Public API ──────────────────────────────────────────────────────────

  /// Returns true if at least one engine is available.
  Future<bool> initialize() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) return false;

    if (!_nativeInitialized) {
      try {
        _nativeInitialized = await _nativeStt.initialize(
          onStatus: _onNativeStatus,
          onError: _onNativeError,
        );
      } catch (e) {
        debugPrint('HybridSpeech: native init failed: $e');
        _nativeInitialized = false;
      }
    }

    try {
      _deepgramAvailable = await DeepgramService.instance.isAvailable();
    } catch (_) {
      _deepgramAvailable = false;
    }

    // Pick best engine
    if (!_nativeInitialized && _deepgramAvailable) {
      _activeEngine = SpeechEngine.deepgram;
    } else {
      _activeEngine = SpeechEngine.native;
    }

    return _nativeInitialized || _deepgramAvailable;
  }

  Future<bool> isAvailable() => initialize();

  Future<void> startListening() async {
    if (_isListening) return;

    await initialize();

    _committedText = '';
    _isListening = true;
    _transcriptCtrl ??= StreamController<HybridTranscript>.broadcast();
    _audioLevelCtrl ??= StreamController<double>.broadcast();

    if (_activeEngine == SpeechEngine.native && _nativeInitialized) {
      await _startNative();
    } else if (_deepgramAvailable) {
      _activeEngine = SpeechEngine.deepgram;
      await _startDeepgram();
    } else {
      _isListening = false;
      _transcriptCtrl?.addError(
        Exception('No speech engine available'),
      );
    }
  }

  Future<String> stopListening() async {
    if (!_isListening) return _committedText.trim();
    _isListening = false;

    if (_activeEngine == SpeechEngine.native) {
      try {
        await _nativeStt.stop();
      } catch (_) {}
    } else {
      try {
        await _deepgramSub?.cancel();
        _deepgramSub = null;
        final text = await DeepgramService.instance.stopAndTranscribe();
        if (text.isNotEmpty && _committedText.isEmpty) {
          _committedText = text;
        }
      } catch (_) {}
    }

    return _committedText.trim();
  }

  Future<void> cancelListening() async {
    _isListening = false;
    _committedText = '';
    try {
      await _nativeStt.stop();
    } catch (_) {}
    try {
      await _deepgramSub?.cancel();
      _deepgramSub = null;
      await DeepgramService.instance.cancelRecording();
    } catch (_) {}
  }

  // ── Native STT ──────────────────────────────────────────────────────────

  Future<void> _startNative() async {
    try {
      await _nativeStt.listen(
        onResult: _onNativeResult,
        onSoundLevelChange: _onNativeSoundLevel,
        listenOptions: SpeechListenOptions(
          partialResults: true,
          listenMode: ListenMode.dictation,
          cancelOnError: false,
        ),
      );
    } catch (e) {
      debugPrint('HybridSpeech: native listen failed: $e');
      // Fall back to Deepgram
      if (_deepgramAvailable) {
        _activeEngine = SpeechEngine.deepgram;
        await _startDeepgram();
      } else {
        _isListening = false;
        _transcriptCtrl?.addError(Exception('Speech recognition failed'));
      }
    }
  }

  void _onNativeResult(SpeechRecognitionResult result) {
    if (!_isListening) return;

    final text = result.recognizedWords.trim();
    if (text.isEmpty) return;

    if (result.finalResult) {
      // Track confidence
      final conf = result.hasConfidenceRating ? result.confidence : 1.0;
      _recentConfidences.add(conf);
      if (_recentConfidences.length > 10) _recentConfidences.removeAt(0);

      // Check if we should switch to Deepgram
      if (conf < _confidenceThreshold && _deepgramAvailable) {
        final lowCount =
            _recentConfidences.where((c) => c < _confidenceThreshold).length;
        if (lowCount >= _lowConfidenceSwitchCount) {
          debugPrint('HybridSpeech: switching to Deepgram (low confidence)');
          _switchToDeepgram();
          return;
        }
      }

      // Accept native result
      if (_committedText.isNotEmpty) {
        _committedText += ' $text';
      } else {
        _committedText = text;
      }

      _transcriptCtrl?.add(HybridTranscript(
        text: _committedText,
        isFinal: true,
        confidence: conf,
        source: SpeechEngine.native,
      ));

      // Re-start for continuous dictation (native auto-stops after each utterance)
      _restartNativeAfterPause();
    } else {
      // Partial — show immediately
      final display =
          _committedText.isEmpty ? text : '$_committedText $text';
      _transcriptCtrl?.add(HybridTranscript(
        text: display,
        isFinal: false,
        confidence: result.hasConfidenceRating ? result.confidence : 1.0,
        source: SpeechEngine.native,
      ));
    }
  }

  void _onNativeSoundLevel(double level) {
    if (!_isListening) return;
    // Native levels are typically -2 to 10 dB — normalize to 0..1
    final normalized = ((level + 2) / 12).clamp(0.0, 1.0);
    _audioLevelCtrl?.add(normalized);
  }

  void _onNativeStatus(String status) {
    debugPrint('HybridSpeech: status=$status');
    if (status == 'done' || status == 'notListening') {
      if (_isListening && _activeEngine == SpeechEngine.native) {
        _restartNativeAfterPause();
      }
    }
  }

  void _onNativeError(SpeechRecognitionError error) {
    debugPrint('HybridSpeech: error=${error.errorMsg}');
    if (error.permanent && _isListening) {
      if (_deepgramAvailable) {
        _switchToDeepgram();
      } else {
        _isListening = false;
        _transcriptCtrl?.addError(
          Exception('Speech recognition error: ${error.errorMsg}'),
        );
      }
    }
  }

  void _restartNativeAfterPause() {
    if (!_isListening || _activeEngine != SpeechEngine.native) return;
    Future.delayed(const Duration(milliseconds: 150), () {
      if (!_isListening || _activeEngine != SpeechEngine.native) return;
      if (!_nativeStt.isListening) {
        _nativeStt
            .listen(
              onResult: _onNativeResult,
              onSoundLevelChange: _onNativeSoundLevel,
              listenOptions: SpeechListenOptions(
                partialResults: true,
                listenMode: ListenMode.dictation,
                cancelOnError: false,
              ),
            )
            .catchError((_) {});
      }
    });
  }

  // ── Deepgram fallback ───────────────────────────────────────────────────

  Future<void> _switchToDeepgram() async {
    try {
      await _nativeStt.stop();
    } catch (_) {}

    _activeEngine = SpeechEngine.deepgram;
    await _startDeepgram();
  }

  Future<void> _startDeepgram() async {
    try {
      _deepgramSub?.cancel();
      _deepgramSub = DeepgramService.instance
          .startRealtimeTranscription()
          .listen(
        (transcript) {
          if (!_isListening) return;
          final text = transcript.text.trim();
          if (text.isEmpty) return;

          if (transcript.isFinal) {
            if (_committedText.isNotEmpty) {
              _committedText += ' $text';
            } else {
              _committedText = text;
            }
          }

          final display = transcript.isFinal
              ? _committedText
              : (_committedText.isEmpty ? text : '$_committedText $text');

          _transcriptCtrl?.add(HybridTranscript(
            text: display,
            isFinal: transcript.isFinal,
            confidence: 0.9,
            source: SpeechEngine.deepgram,
          ));
        },
        onError: (Object error) {
          if (_isListening) {
            _transcriptCtrl?.addError(error);
          }
        },
      );

      // Simulate audio levels from Deepgram (no native sound levels available)
      _simulateDeepgramAudioLevels();
    } catch (e) {
      _isListening = false;
      _transcriptCtrl?.addError(
        Exception('Deepgram failed: $e'),
      );
    }
  }

  Timer? _deepgramLevelTimer;

  void _simulateDeepgramAudioLevels() {
    _deepgramLevelTimer?.cancel();
    final rng = math.Random();
    _deepgramLevelTimer = Timer.periodic(
      const Duration(milliseconds: 50),
      (_) {
        if (!_isListening || _activeEngine != SpeechEngine.deepgram) {
          _deepgramLevelTimer?.cancel();
          return;
        }
        // Generate organic-feeling pseudo levels
        final base = 0.15 + rng.nextDouble() * 0.45;
        _audioLevelCtrl?.add(base);
      },
    );
  }

  // ── Compute RMS from PCM data (utility for future use) ─────────────────

  static double computeRmsLevel(Uint8List pcmData) {
    if (pcmData.length < 2) return 0.0;
    final samples = Int16List.view(pcmData.buffer);
    double sum = 0;
    for (final s in samples) {
      sum += s * s;
    }
    final rms = math.sqrt(sum / samples.length);
    return (rms / 32768.0).clamp(0.0, 1.0);
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  void dispose() {
    _deepgramLevelTimer?.cancel();
    _transcriptCtrl?.close();
    _transcriptCtrl = null;
    _audioLevelCtrl?.close();
    _audioLevelCtrl = null;
    _deepgramSub?.cancel();
  }
}
