// lib/services/native_resources_service.dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class NativeResourcesService {
  static final NativeResourcesService _instance =
      NativeResourcesService._internal();
  factory NativeResourcesService() => _instance;
  NativeResourcesService._internal();

  String? _resourcesPath;
  Map<String, dynamic>? _masterIndex;

  Future<String> get resourcesPath async {
    if (_resourcesPath != null) return _resourcesPath!;
    final appDir = await getApplicationDocumentsDirectory();
    _resourcesPath = '${appDir.path}/../../native_resources';
    return _resourcesPath!;
  }

  Future<Map<String, dynamic>> get masterIndex async {
    if (_masterIndex != null) return _masterIndex!;

    final path = await resourcesPath;
    final indexFile = File('$path/master_index.json');

    if (await indexFile.exists()) {
      final content = await indexFile.readAsString();
      _masterIndex = Map<String, dynamic>.from(
        content.isNotEmpty ? {} : {},
      );
    }

    return _masterIndex ?? {};
  }

  Future<bool> hasResources(String code) async {
    final index = await masterIndex;
    return index['subjects']?[code] != null;
  }

  Future<List<String>> getDownloadedSources(String code) async {
    final index = await masterIndex;
    final subject = index['subjects']?[code];
    if (subject == null) return [];
    return List<String>.from(subject['downloaded_sources'] ?? []);
  }

  Future<String?> getResourcePath(
      String code, String source, String filename) async {
    final path = await resourcesPath;
    final level = _getLevelForCode(code);

    final filePath = '$path/$level/$code/$source/$filename';
    final file = File(filePath);

    if (await file.exists()) {
      return filePath;
    }
    return null;
  }

  Future<List<Map<String, String>>> listResourceFiles(String code) async {
    final index = await masterIndex;
    final subject = index['subjects']?[code];
    if (subject == null) return [];

    final level = _getLevelForCode(code);
    final path = await resourcesPath;
    final files = <Map<String, String>>[];

    for (final source
        in List<String>.from(subject['downloaded_sources'] ?? [])) {
      final sourceDir = Directory('$path/$level/$code/$source');
      if (await sourceDir.exists()) {
        await for (final entity in sourceDir.list()) {
          if (entity is File && entity.path.endsWith('.html')) {
            files.add({
              'source': source,
              'path': entity.path,
              'name': entity.path.split(Platform.pathSeparator).last,
            });
          }
        }
      }
    }

    return files;
  }

  String _getLevelForCode(String code) {
    // IGCSE codes (4 digits, 0xxx)
    if (code.startsWith('0') && int.tryParse(code.substring(0, 1)) != null) {
      return 'IGCSE';
    }
    // AS/A Level codes (9xxx)
    if (code.startsWith('9')) {
      return 'AS-A';
    }
    // O Level codes (5xxx)
    if (code.startsWith('5')) {
      return 'OLevel';
    }
    return 'IGCSE';
  }

  Future<void> scanForResources() async {
    final path = await resourcesPath;
    final dir = Directory(path);

    if (!await dir.exists()) {
      return;
    }

    final subjects = <String, dynamic>{};

    await for (final levelDir in dir.list()) {
      if (levelDir is Directory) {
        final level = levelDir.path.split(Platform.pathSeparator).last;

        await for (final subjectDir in levelDir.list()) {
          if (subjectDir is Directory) {
            final code = subjectDir.path.split(Platform.pathSeparator).last;

            final sources = <String>[];
            await for (final sourceDir in subjectDir.list()) {
              if (sourceDir is Directory) {
                sources.add(sourceDir.path.split(Platform.pathSeparator).last);
              }
            }

            if (sources.isNotEmpty) {
              subjects[code] = {
                'level': level,
                'downloaded_sources': sources,
              };
            }
          }
        }
      }
    }

    _masterIndex = {
      'subjects': subjects,
      'scanned_at': DateTime.now().toIso8601String(),
    };
  }
}

final nativeResourcesServiceProvider = Provider<NativeResourcesService>((ref) {
  return NativeResourcesService();
});

// Provider to check if resources exist for a subject
final hasNativeResourcesProvider =
    FutureProvider.family<bool, String>((ref, code) async {
  final service = ref.read(nativeResourcesServiceProvider);
  return service.hasResources(code);
});

// Provider to list downloaded resources
final nativeResourcesListProvider =
    FutureProvider.family<List<Map<String, String>>, String>((ref, code) async {
  final service = ref.read(nativeResourcesServiceProvider);
  return service.listResourceFiles(code);
});
