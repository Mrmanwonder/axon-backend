import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

class VisionRuntimeCapabilities {
  const VisionRuntimeCapabilities({
    required this.supportsPythonOffline,
    required this.supportsMobileFallback,
    required this.embeddedPython,
    required this.platform,
  });

  final bool supportsPythonOffline;
  final bool supportsMobileFallback;
  final bool embeddedPython;
  final String platform;

  factory VisionRuntimeCapabilities.fromMap(Map<String, dynamic> map) {
    return VisionRuntimeCapabilities(
      supportsPythonOffline: map['supports_python_offline'] == true,
      supportsMobileFallback: map['supports_mobile_fallback'] != false,
      embeddedPython: map['embedded_python'] == true,
      platform: (map['platform'] ?? 'unknown').toString(),
    );
  }

  Map<String, dynamic> toMap() => {
        'supports_python_offline': supportsPythonOffline,
        'supports_mobile_fallback': supportsMobileFallback,
        'embedded_python': embeddedPython,
        'platform': platform,
      };
}

class DocumentVisionRuntime {
  const DocumentVisionRuntime({MethodChannel? channel})
      : _channel = channel ??
            const MethodChannel('com.axon.app/utils');

  final MethodChannel _channel;

  Future<VisionRuntimeCapabilities> getCapabilities() async {
    try {
      final raw = await _channel.invokeMethod('getVisionRuntimeCapabilities');
      if (raw is Map) {
        return VisionRuntimeCapabilities.fromMap(
          Map<String, dynamic>.from(raw),
        );
      }
    } catch (_) {
      // Fall through to platform heuristics below.
    }

    final isMobile = Platform.isAndroid || Platform.isIOS;
    return VisionRuntimeCapabilities(
      supportsPythonOffline: !isMobile,
      supportsMobileFallback: true,
      embeddedPython: false,
      platform: Platform.operatingSystem,
    );
  }

  Future<Map<String, dynamic>?> runPythonOffline(String filePath) async {
    if (Platform.isAndroid || Platform.isIOS) {
      return null;
    }

    if (RegExp(r'[;\'"|`$]').hasMatch(filePath)) return null;

    final commands = Platform.isWindows
        ? const [
            ['py', '-3'],
            ['python'],
          ]
        : const [
            ['python3'],
            ['python'],
          ];

    for (final command in commands) {
      try {
        final result = await Process.run(
          command.first,
          [
            ...command.skip(1),
            '-m',
            'tools.document_vision.cli',
            '--input',
            filePath,
          ],
          runInShell: Platform.isWindows,
        );
        if (result.exitCode != 0) {
          continue;
        }
        final stdout = (result.stdout ?? '').toString().trim();
        if (stdout.isEmpty) {
          continue;
        }
        final decoded = jsonDecode(stdout);
        if (decoded is Map) {
          return Map<String, dynamic>.from(decoded);
        }
      } catch (_) {
        continue;
      }
    }
    return null;
  }
}
