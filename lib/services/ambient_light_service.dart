import 'dart:async';
import 'package:axon/services/personalization_service.dart';

class AmbientLightService {
  static final AmbientLightService _instance = AmbientLightService._internal();
  factory AmbientLightService() => _instance;
  AmbientLightService._internal();

  bool _isMonitoring = false;

  Future<void> startMonitoring() async {
    if (_isMonitoring) return;
    _isMonitoring = true;
  }

  void stopMonitoring() {
    _isMonitoring = false;
  }

  void dismissSuggestion() {
    greyscaleModeNotifierProvider.dismissSuggestion();
  }
}

final ambientLightService = AmbientLightService();