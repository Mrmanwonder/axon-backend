import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'study_resources_service.dart';

class ResourceDownloadService {
  static final ResourceDownloadService _instance =
      ResourceDownloadService._internal();
  factory ResourceDownloadService() => _instance;
  ResourceDownloadService._internal();

  static const String _progressKey = 'resource_download_progress';
  static const String _lastSyncKey = 'resource_last_sync';

  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String _currentSubject = '';
  String _currentResource = '';

  bool get isDownloading => _isDownloading;
  double get downloadProgress => _downloadProgress;
  String get currentSubject => _currentSubject;
  String get currentResource => _currentResource;

  Future<void> downloadAllResources({
    required Function(double progress, String subject, String resource)
        onProgress,
    required Function(String subject, String error) onError,
  }) async {
    if (_isDownloading) return;
    _isDownloading = true;
    _downloadProgress = 0.0;

    try {
      final service = StudyResourcesService();
      final allCodes = service.getAllCodes();
      final totalResources = _countTotalResources(service);
      int processedResources = 0;

      for (final code in allCodes) {
        final resources = service.getResources(code);
        if (resources == null) continue;

        _currentSubject = '${resources.name} (${resources.code})';

        await _downloadResourceList(
          resources.notesUrls,
          'notes',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        await _downloadResourceList(
          resources.videoUrls,
          'videos',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        await _downloadResourceList(
          resources.flashcardsUrls,
          'flashcards',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        await _downloadResourceList(
          resources.examPrepUrls,
          'exam_prep',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        await _downloadResourceList(
          resources.pastPaperUrls,
          'past_papers',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        await _downloadResourceList(
          resources.otherUrls,
          'other',
          code,
          onProgress,
          onError,
          (count) => processedResources += count,
        );

        _downloadProgress =
            totalResources > 0 ? processedResources / totalResources : 0.0;
        onProgress(
            _downloadProgress, _currentSubject, 'Completed ${resources.name}');
      }

      await _saveLastSync();
    } catch (e) {
      debugPrint('ResourceDownloadService error: $e');
      onError(_currentSubject, e.toString());
    } finally {
      _isDownloading = false;
      _downloadProgress = 1.0;
    }
  }

  Future<void> _downloadResourceList(
    List<String> urls,
    String type,
    String code,
    Function(double, String, String) onProgress,
    Function(String, String) onError,
    Function(int) onProcessed,
  ) async {
    if (urls.isEmpty) {
      onProcessed(0);
      return;
    }

    final appDir = await getApplicationDocumentsDirectory();
    final resourceDir =
        Directory('${appDir.path}/cached_resources/$code/$type');
    if (!await resourceDir.exists()) {
      await resourceDir.create(recursive: true);
    }

    int processed = 0;
    for (final url in urls) {
      if (!_isDownloading) break;

      _currentResource = url;
      try {
        final fileName = _sanitizeFileName(url);
        final file = File('${resourceDir.path}/$fileName');

        if (!await file.exists()) {
          await _downloadSingleFile(url, file.path);
        }

        await _saveProgressEntry(code, type, url);

        processed++;
        onProcessed(processed);
        onProgress(_downloadProgress, _currentSubject, 'Downloading: $url');

        await Future.delayed(const Duration(milliseconds: 500));
      } catch (e) {
        debugPrint('Error downloading $url: $e');
        onError(_currentSubject, 'Failed to download $url: $e');
      }
    }
    onProcessed(processed);
  }

  Future<void> _downloadSingleFile(String url, String savePath) async {
    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final file = File(savePath);
        await file.writeAsBytes(response.bodyBytes);
      }
    } catch (e) {
      debugPrint('Download error for $url: $e');
    }
  }

  String _sanitizeFileName(String url) {
    final uri = Uri.tryParse(url);
    final fileName = uri?.pathSegments.isNotEmpty == true
        ? uri!.pathSegments.last
        : 'resource_${DateTime.now().millisecondsSinceEpoch}';
    return fileName.replaceAll(RegExp(r'[^\w\-.]'), '_').substring(0, 50);
  }

  int _countTotalResources(StudyResourcesService service) {
    int count = 0;
    for (final code in service.getAllCodes()) {
      final resources = service.getResources(code);
      if (resources != null) {
        count += resources.notesUrls.length;
        count += resources.videoUrls.length;
        count += resources.flashcardsUrls.length;
        count += resources.examPrepUrls.length;
        count += resources.pastPaperUrls.length;
        count += resources.otherUrls.length;
      }
    }
    return count;
  }

  Future<void> _saveProgressEntry(String code, String type, String url) async {
    final prefs = await SharedPreferences.getInstance();
    final progress = prefs.getStringList(_progressKey) ?? [];
    progress.add('$code|$type|$url');
    await prefs.setStringList(_progressKey, progress);
  }

  Future<void> _saveLastSync() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_lastSyncKey, DateTime.now().millisecondsSinceEpoch);
  }

  Future<DateTime?> getLastSyncTime() async {
    final prefs = await SharedPreferences.getInstance();
    final timestamp = prefs.getInt(_lastSyncKey);
    if (timestamp != null) {
      return DateTime.fromMillisecondsSinceEpoch(timestamp);
    }
    return null;
  }

  Future<void> clearCache() async {
    final appDir = await getApplicationDocumentsDirectory();
    final resourceDir = Directory('${appDir.path}/cached_resources');
    if (await resourceDir.exists()) {
      await resourceDir.delete(recursive: true);
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_progressKey);
    await prefs.remove(_lastSyncKey);
  }

  Future<int> getCachedResourceCount() async {
    final appDir = await getApplicationDocumentsDirectory();
    final resourceDir = Directory('${appDir.path}/cached_resources');
    if (!await resourceDir.exists()) return 0;

    int count = 0;
    await for (final entity in resourceDir.list(recursive: true)) {
      if (entity is File) count++;
    }
    return count;
  }

  Future<bool> isResourceCached(String code, String url) async {
    final prefs = await SharedPreferences.getInstance();
    final progress = prefs.getStringList(_progressKey) ?? [];
    return progress.any((p) => p.startsWith('$code|') && p.contains(url));
  }

  Future<void> downloadSubjectResources(
    String code, {
    required Function(double, String) onProgress,
    Function(String)? onError,
  }) async {
    final service = StudyResourcesService();
    final resources = service.getResources(code);
    if (resources == null) return;

    _currentSubject = resources.name;

    final allUrls = <String>[
      ...resources.notesUrls,
      ...resources.videoUrls,
      ...resources.flashcardsUrls,
      ...resources.examPrepUrls,
      ...resources.pastPaperUrls,
      ...resources.otherUrls,
    ];

    final totalUrls = allUrls.length;
    int processed = 0;

    for (final url in allUrls) {
      _currentResource = url;
      try {
        final appDir = await getApplicationDocumentsDirectory();
        final fileName = _sanitizeFileName(url);
        final file = File('${appDir.path}/cached_resources/$code/$fileName');

        if (!await file.exists()) {
          await file.parent.create(recursive: true);
          await _downloadSingleFile(url, file.path);
        }

        processed++;
        onProgress(
            totalUrls > 0 ? processed / totalUrls : 0.0, 'Downloading $url');
      } catch (e) {
        onError?.call('Failed to download $url: $e');
      }
    }
  }

  void cancelDownload() {
    _isDownloading = false;
  }
}

final resourceDownloadServiceProvider = ResourceDownloadService();
