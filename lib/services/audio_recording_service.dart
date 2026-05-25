import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

class AudioRecordingService {
  static final AudioRecordingService _instance =
      AudioRecordingService._internal();
  factory AudioRecordingService() => _instance;
  AudioRecordingService._internal();

  bool _isRecording = false;
  StreamController<Uint8List>? _audioStreamController;

  Future<void> initialize() async {}

  Future<bool> isRecordingAvailable() async {
    if (kIsWeb) return true;
    final micStatus = await Permission.microphone.status;
    return micStatus.isGranted;
  }

  Future<bool> requestMicrophonePermission() async {
    if (kIsWeb) return true;
    final status = await Permission.microphone.request();
    return status.isGranted;
  }

  Stream<Uint8List> startRecording() {
    if (_audioStreamController?.isClosed == false) {
      return _audioStreamController!.stream;
    }

    _audioStreamController = StreamController<Uint8List>();
    _isRecording = true;
    return _audioStreamController!.stream;
  }

  Future<void> stopRecording() async {
    _isRecording = false;
    await _audioStreamController?.close();
    _audioStreamController = null;
  }

  bool get isRecording => _isRecording;

  Future<void> dispose() async {
    await stopRecording();
  }
}
