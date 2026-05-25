import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:web_socket_channel/io.dart';

import 'secure_credentials_service.dart';

class DeepgramTranscript {
  final String text;
  final bool isFinal;
  final bool speechFinal;

  const DeepgramTranscript({
    required this.text,
    required this.isFinal,
    required this.speechFinal,
  });
}

class DeepgramService {
  DeepgramService._();

  static final DeepgramService instance = DeepgramService._();
  static const MethodChannel _utilsChannel =
      MethodChannel('com.axon.app/utils');
  static const int _sampleRate = 16000;
  static const int _channels = 1;

  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();

  String? _apiKey;
  bool _recorderOpened = false;
  bool _isRecording = false;
  IOWebSocketChannel? _liveChannel;
  StreamController<Uint8List>? _audioController;
  StreamController<DeepgramTranscript>? _liveTranscriptController;
  StreamSubscription<Uint8List>? _audioSubscription;
  StreamSubscription<dynamic>? _socketSubscription;
  String _latestLiveTranscript = '';

  bool get isRecording => _isRecording;
  String get latestLiveTranscript => _latestLiveTranscript;

  Future<bool> isAvailable() async {
    try {
      final apiKey = await _resolveApiKey();
      return apiKey.isNotEmpty;
    } catch (_) {
      return false;
    }
  }

  Stream<DeepgramTranscript> startRealtimeTranscription() async* {
    if (_isRecording) {
      yield* _liveTranscriptController?.stream ?? const Stream.empty();
      return;
    }

    await _startLiveSession();
    yield* _liveTranscriptController!.stream;
  }

  Future<void> startRecording() => _startLiveSession();

  Future<String> stopAndTranscribe() async {
    await stopLiveTranscription();
    return _latestLiveTranscript.trim();
  }

  Future<void> stopLiveTranscription() async {
    final wasRecording = _isRecording;
    _isRecording = false;

    if (wasRecording) {
      try {
        await _recorder.stopRecorder();
      } catch (_) {}
    }

    await _audioSubscription?.cancel();
    _audioSubscription = null;

    await _audioController?.close();
    _audioController = null;

    try {
      _liveChannel?.sink.add(jsonEncode({'type': 'CloseStream'}));
    } catch (_) {}

    await _socketSubscription?.cancel();
    _socketSubscription = null;

    await _liveChannel?.sink.close();
    _liveChannel = null;

    await _liveTranscriptController?.close();
    _liveTranscriptController = null;
  }

  Future<void> cancelRecording() async {
    await stopLiveTranscription();
  }

  Future<String> transcribeAudioFile(String filePath) async {
    final apiKey = await _resolveApiKey();
    if (apiKey.isEmpty) {
      throw const DeepgramException('Deepgram API key is not configured.');
    }

    final file = File(filePath);
    if (!await file.exists()) {
      throw DeepgramException('Audio file not found: $filePath');
    }

    try {
      final response = await http
          .post(
            _restUri,
            headers: {
              HttpHeaders.authorizationHeader: 'Token $apiKey',
              HttpHeaders.contentTypeHeader: 'audio/wav',
            },
            body: await file.readAsBytes(),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw DeepgramException(
          'Deepgram transcription failed (${response.statusCode}).',
        );
      }

      final payload = jsonDecode(response.body) as Map<String, dynamic>;
      final transcript = _extractRestTranscript(payload);
      if (transcript.isEmpty) {
        throw const DeepgramException('No speech was detected.');
      }

      return transcript;
    } on TimeoutException {
      throw const DeepgramException('Deepgram transcription timed out.');
    } finally {
      unawaited(file.delete().catchError((_) => file));
    }
  }

  Stream<String> startLiveTranscription() {
    return startRealtimeTranscription().map((event) => event.text);
  }

  Future<void> _startLiveSession() async {
    final apiKey = await _resolveApiKey();
    if (apiKey.isEmpty) {
      throw const DeepgramException('Deepgram API key is not configured.');
    }

    final hasPermission = await _requestMicrophonePermission();
    if (!hasPermission) {
      throw const DeepgramException('Microphone permission was denied.');
    }

    await _openRecorder();
    await stopLiveTranscription();

    _latestLiveTranscript = '';
    _liveTranscriptController =
        StreamController<DeepgramTranscript>.broadcast();
    _audioController = StreamController<Uint8List>();
    _liveChannel = IOWebSocketChannel.connect(
      _liveUri,
      headers: {HttpHeaders.authorizationHeader: 'Token $apiKey'},
      pingInterval: const Duration(seconds: 20),
      connectTimeout: const Duration(seconds: 10),
    );

    await _liveChannel!.ready;

    _socketSubscription = _liveChannel!.stream.listen(
      _handleLiveMessage,
      onError: (Object error) {
        _liveTranscriptController?.addError(
          DeepgramException('Deepgram live stream failed: $error'),
        );
      },
      onDone: () {
        _isRecording = false;
      },
    );

    _audioSubscription = _audioController!.stream.listen(
      (chunk) {
        if (_isRecording && chunk.isNotEmpty) {
          _liveChannel?.sink.add(chunk);
        }
      },
      onError: (Object error) {
        _liveTranscriptController?.addError(error);
      },
    );

    await _recorder.startRecorder(
      codec: Codec.pcm16,
      toStream: _audioController!.sink,
      sampleRate: _sampleRate,
      numChannels: _channels,
      bitRate: 256000,
      bufferSize: 4096,
      enableNoiseSuppression: true,
      enableEchoCancellation: true,
      audioSource: AudioSource.microphone,
    );

    _isRecording = true;
  }

  void _handleLiveMessage(dynamic message) {
    if (message is! String) return;

    final payload = jsonDecode(message) as Map<String, dynamic>;
    if (payload['type'] != 'Results') return;

    final transcript = _extractLiveTranscript(payload);
    if (transcript.isEmpty) return;

    final event = DeepgramTranscript(
      text: transcript,
      isFinal: payload['is_final'] == true,
      speechFinal: payload['speech_final'] == true,
    );

    _latestLiveTranscript = transcript;
    _liveTranscriptController?.add(event);
  }

  Future<String> _resolveApiKey() async {
    if (_isUsableApiKey(_apiKey)) return _apiKey!;

    // Fetch from backend via SecureCredentialsService
    try {
      final creds = SecureCredentialsService();
      final allCreds = await creds.getAllCredentials();
      final key = allCreds.effectiveDeepgramKey?.trim();
      if (_isUsableApiKey(key)) {
        _apiKey = key;
        return _apiKey!;
      }
    } catch (_) {}

    // Fallback: Native platform channel for key stored during build
    try {
      const channel = MethodChannel('com.axon.app/utils');
      final result = await channel.invokeMethod<String>('getDeepgramApiKey');
      if (_isUsableApiKey(result)) {
        _apiKey = result!.trim();
        return _apiKey!;
      }
    } catch (_) {}

    _apiKey = '';
    return '';
  }

  Future<bool> _requestMicrophonePermission() async {
    if (kIsWeb) return true;
    final status = await Permission.microphone.request();
    return status.isGranted;
  }

  Future<void> _openRecorder() async {
    if (_recorderOpened) return;
    await _recorder.openRecorder();
    _recorderOpened = true;
  }

  Uri get _liveUri {
    return Uri.https(
      'api.deepgram.com',
      '/v1/listen',
      const {
        'model': 'nova-2',
        'encoding': 'linear16',
        'sample_rate': '16000',
        'channels': '1',
        'interim_results': 'true',
        'endpointing': '300',
        'utterance_end_ms': '1000',
        'vad_events': 'true',
        'smart_format': 'true',
        'punctuate': 'true',
        'language': 'en',
      },
    );
  }

  Uri get _restUri {
    return Uri.https(
      'api.deepgram.com',
      '/v1/listen',
      const {
        'model': 'nova-2',
        'smart_format': 'true',
        'punctuate': 'true',
        'language': 'en',
      },
    );
  }

  String _extractLiveTranscript(Map<String, dynamic> payload) {
    final channel = payload['channel'];
    if (channel is! Map) return '';
    return _extractTranscriptFromChannel(channel);
  }

  String _extractRestTranscript(Map<String, dynamic> payload) {
    final results = payload['results'];
    if (results is! Map) return '';

    final channels = results['channels'];
    if (channels is! List || channels.isEmpty) return '';

    final firstChannel = channels.first;
    if (firstChannel is! Map) return '';
    return _extractTranscriptFromChannel(firstChannel);
  }

  String _extractTranscriptFromChannel(Map<dynamic, dynamic> channel) {
    final alternatives = channel['alternatives'];
    if (alternatives is! List || alternatives.isEmpty) return '';

    final firstAlternative = alternatives.first;
    if (firstAlternative is! Map) return '';

    return (firstAlternative['transcript'] ?? '').toString().trim();
  }

  bool _isUsableApiKey(String? value) {
    final key = value?.trim();
    return key != null && key.isNotEmpty;
  }
}

class DeepgramException implements Exception {
  final String message;

  const DeepgramException(this.message);

  @override
  String toString() => message;
}
