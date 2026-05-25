import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AxonHaptics {
  static const String _prefsKey = 'haptics_enabled';
  static bool _enabled = true;
  static bool _loaded = false;

  static Future<void> ensureLoaded() async {
    if (_loaded) return;
    final prefs = await SharedPreferences.getInstance();
    _enabled = prefs.getBool(_prefsKey) ?? true;
    _loaded = true;
  }

  static Future<bool> isEnabled() async {
    await ensureLoaded();
    return _enabled;
  }

  static Future<void> setEnabled(bool value) async {
    _enabled = value;
    _loaded = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefsKey, value);
  }

  static Future<void> lightImpact() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.lightImpact();
  }

  static Future<void> mediumImpact() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.mediumImpact();
  }

  static Future<void> heavyImpact() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.heavyImpact();
  }

  static Future<void> selectionClick() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.selectionClick();
  }

  static Future<void> success() async {
    await rewardHarmonic();
  }

  static Future<void> rewardHarmonic() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.mediumImpact();
    await Future<void>.delayed(const Duration(milliseconds: 70));
    await HapticFeedback.heavyImpact();
    await Future<void>.delayed(const Duration(milliseconds: 90));
    await HapticFeedback.mediumImpact();
  }

  static Future<void> alertStaccato() async {
    await ensureLoaded();
    if (!_enabled) return;
    await HapticFeedback.selectionClick();
    await Future<void>.delayed(const Duration(milliseconds: 55));
    await HapticFeedback.selectionClick();
  }
}
