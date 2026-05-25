import 'dart:io';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

class OverlayPermissionService {
  static final OverlayPermissionService _instance = OverlayPermissionService._internal();
  factory OverlayPermissionService() => _instance;
  OverlayPermissionService._internal();

  static const _channel = MethodChannel('com.axon.studylock');

  Future<bool> hasOverlayPermission() async {
    if (!Platform.isAndroid) return true;

    try {
      final result = await _channel.invokeMethod<bool>('hasOverlayPermission');
      return result ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<bool> requestOverlayPermission() async {
    if (!Platform.isAndroid) return true;

    try {
      final status = await Permission.systemAlertWindow.request();
      return status.isGranted;
    } catch (e) {
      return false;
    }
  }

  Future<bool> canRequestOverlayPermission() async {
    if (!Platform.isAndroid) return false;

    try {
      return await Permission.systemAlertWindow.shouldShowRequestRationale;
    } catch (e) {
      return true;
    }
  }

  Future<void> openOverlaySettings() async {
    if (!Platform.isAndroid) return;

    try {
      await _channel.invokeMethod('openOverlaySettings');
    } catch (e) {
      await openAppSettings();
    }
  }

  Future<bool> isAccessibilityServiceEnabled() async {
    if (!Platform.isAndroid) return false;

    try {
      final result = await _channel.invokeMethod<bool>('isAccessibilityEnabled');
      return result ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<void> openAccessibilitySettings() async {
    if (!Platform.isAndroid) return;

    try {
      await _channel.invokeMethod('requestAccessibilityPermission');
    } catch (e) {
      await openAppSettings();
    }
  }

  Future<bool> checkAndRequestOverlayPermission() async {
    if (!Platform.isAndroid) return true;

    final hasPermission = await hasOverlayPermission();
    if (hasPermission) return true;

    final canRequest = await canRequestOverlayPermission();
    if (canRequest) {
      return await requestOverlayPermission();
    } else {
      await openOverlaySettings();
      return false;
    }
  }
}

final overlayPermissionService = OverlayPermissionService();