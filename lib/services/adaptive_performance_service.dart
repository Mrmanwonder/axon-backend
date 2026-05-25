// lib/services/adaptive_performance_service.dart
//
// Adaptive performance service that detects device capabilities
// and adjusts app behavior accordingly for optimal performance

import 'dart:io';
import 'package:flutter/foundation.dart';

enum DeviceTier {
  low, // < 3GB RAM, old dual-core
  mid, // 3-6GB RAM, mid-range
  high, // > 6GB RAM, flagship
}

class DeviceCapabilities {
  final DeviceTier tier;
  final int totalMemoryMB;
  final int cores;
  final bool isLowEnd;
  final double imageQuality; // 0.0 - 1.0
  final int maxCacheSize; // items
  final int imageMaxDimension; // pixels
  final bool enableAnimations;
  final bool enableParallax;
  final int chunkSizeMB; // for model loading
  final bool preloadImages;

  const DeviceCapabilities({
    required this.tier,
    required this.totalMemoryMB,
    required this.cores,
    required this.isLowEnd,
    required this.imageQuality,
    required this.maxCacheSize,
    required this.imageMaxDimension,
    required this.enableAnimations,
    required this.enableParallax,
    required this.chunkSizeMB,
    required this.preloadImages,
  });

  static const DeviceCapabilities low = DeviceCapabilities(
    tier: DeviceTier.low,
    totalMemoryMB: 2048,
    cores: 2,
    isLowEnd: true,
    imageQuality: 0.5,
    maxCacheSize: 50,
    imageMaxDimension: 800,
    enableAnimations: false,
    enableParallax: false,
    chunkSizeMB: 8,
    preloadImages: false,
  );

  static const DeviceCapabilities mid = DeviceCapabilities(
    tier: DeviceTier.mid,
    totalMemoryMB: 4096,
    cores: 4,
    isLowEnd: false,
    imageQuality: 0.7,
    maxCacheSize: 200,
    imageMaxDimension: 1200,
    enableAnimations: true,
    enableParallax: true,
    chunkSizeMB: 16,
    preloadImages: true,
  );

  static const DeviceCapabilities high = DeviceCapabilities(
    tier: DeviceTier.high,
    totalMemoryMB: 8192,
    cores: 8,
    isLowEnd: false,
    imageQuality: 0.9,
    maxCacheSize: 500,
    imageMaxDimension: 2048,
    enableAnimations: true,
    enableParallax: true,
    chunkSizeMB: 32,
    preloadImages: true,
  );
}

class AdaptivePerformanceService {
  static final AdaptivePerformanceService _instance =
      AdaptivePerformanceService._internal();
  factory AdaptivePerformanceService() => _instance;
  AdaptivePerformanceService._internal();

  DeviceCapabilities? _capabilities;

  DeviceCapabilities get capabilities {
    _capabilities ??= _detectCapabilities();
    return _capabilities!;
  }

  DeviceTier get tier => capabilities.tier;
  bool get isLowEnd => capabilities.isLowEnd;
  bool get isHighEnd => capabilities.tier == DeviceTier.high;

  // Call this early in main.dart before runApp
  DeviceCapabilities initialize() {
    _capabilities ??= _detectCapabilities();
    _applyPerformanceSettings();
    debugPrint(
        'AdaptivePerformance: Device tier = ${_capabilities!.tier.name}, '
        'RAM = ${_capabilities!.totalMemoryMB}MB, cores = ${_capabilities!.cores}');
    return _capabilities!;
  }

  DeviceCapabilities _detectCapabilities() {
    if (kIsWeb) {
      return DeviceCapabilities.mid;
    }

    try {
      if (Platform.isAndroid || Platform.isIOS) {
        final memoryInfo = _getMemoryInfo();
        final cores = _getCoreCount();
        final totalMB = memoryInfo['totalMB'] ?? 3072;
        final isLowEnd = memoryInfo['isLowEnd'] ?? (totalMB < 3000);

        if (isLowEnd || totalMB < 3000) {
          return DeviceCapabilities(
            tier: DeviceTier.low,
            totalMemoryMB: totalMB,
            cores: cores,
            isLowEnd: true,
            imageQuality: 0.5,
            maxCacheSize: 50,
            imageMaxDimension: 800,
            enableAnimations: false,
            enableParallax: false,
            chunkSizeMB: 8,
            preloadImages: false,
          );
        } else if (totalMB < 6000) {
          return DeviceCapabilities(
            tier: DeviceTier.mid,
            totalMemoryMB: totalMB,
            cores: cores,
            isLowEnd: false,
            imageQuality: 0.7,
            maxCacheSize: 200,
            imageMaxDimension: 1200,
            enableAnimations: true,
            enableParallax: true,
            chunkSizeMB: 16,
            preloadImages: true,
          );
        } else {
          return DeviceCapabilities(
            tier: DeviceTier.high,
            totalMemoryMB: totalMB,
            cores: cores,
            isLowEnd: false,
            imageQuality: 0.9,
            maxCacheSize: 500,
            imageMaxDimension: 2048,
            enableAnimations: true,
            enableParallax: true,
            chunkSizeMB: 32,
            preloadImages: true,
          );
        }
      }
    } catch (e) {
      debugPrint(
          'AdaptivePerformance: Detection failed, using mid-tier defaults');
    }

    return DeviceCapabilities.mid;
  }

  Map<String, dynamic> _getMemoryInfo() {
    try {
      if (Platform.isAndroid) {
        final procFile = File('/proc/meminfo');
        if (procFile.existsSync()) {
          final content = procFile.readAsStringSync();
          final memTotal = _extractMemTotal(content);
          if (memTotal > 0) {
            final totalMB = memTotal ~/ 1024;
            return {
              'totalMB': totalMB,
              'isLowEnd': totalMB < 3000,
            };
          }
        }
      }
    } catch (_) {}
    return {'totalMB': 4096, 'isLowEnd': false};
  }

  int _extractMemTotal(String content) {
    try {
      final lines = content.split('\n');
      for (final line in lines) {
        if (line.startsWith('MemTotal')) {
          final parts = line.split(RegExp(r'\s+'));
          if (parts.length >= 2) {
            return int.tryParse(parts[1]) ?? 0;
          }
        }
      }
    } catch (_) {}
    return 0;
  }

  int _getCoreCount() {
    try {
      return Platform.numberOfProcessors;
    } catch (_) {
      return 4;
    }
  }

  void _applyPerformanceSettings() {
    final caps = capabilities;

    // Reduce image quality on low-end devices to save memory
    if (caps.isLowEnd) {
      // Lower Flutter GPU settings
      debugPrint('AdaptivePerformance: Low-end mode - reducing GPU memory');
    }
  }

  // Configurable thresholds per tier
  int get maxConcurrentImageLoads {
    switch (tier) {
      case DeviceTier.low:
        return 2;
      case DeviceTier.mid:
        return 4;
      case DeviceTier.high:
        return 8;
    }
  }

  Duration get animationDuration {
    switch (tier) {
      case DeviceTier.low:
        return Duration.zero; // No animations
      case DeviceTier.mid:
        return const Duration(milliseconds: 200);
      case DeviceTier.high:
        return const Duration(milliseconds: 300);
    }
  }

  Duration get splashDuration {
    switch (tier) {
      case DeviceTier.low:
        return const Duration(milliseconds: 1500);
      case DeviceTier.mid:
        return const Duration(milliseconds: 2000);
      case DeviceTier.high:
        return const Duration(milliseconds: 2500);
    }
  }

  // Image compression quality (0.0-100.0 for dart:ui)
  int get imageCompressionQuality {
    return (capabilities.imageQuality * 100).round();
  }

  // Max dimensions for image resizing before upload
  int get imageMaxUploadDimension {
    return capabilities.imageMaxDimension;
  }

  bool get useListRecycling => !capabilities.isLowEnd;

  // Audio pool size
  int get maxAudioPlayers {
    switch (tier) {
      case DeviceTier.low:
        return 2;
      case DeviceTier.mid:
        return 4;
      case DeviceTier.high:
        return 8;
    }
  }

  // Database page size for SQLite
  int get sqlitePageSize {
    switch (tier) {
      case DeviceTier.low:
        return 1024; // Smaller pages = less memory
      case DeviceTier.mid:
        return 2048;
      case DeviceTier.high:
        return 4096;
    }
  }
}
