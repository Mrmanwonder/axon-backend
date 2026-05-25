import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:googleapis_auth/auth_io.dart' as auth;
import 'package:googleapis/drive/v3.dart' as drive;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'past_paper_service.dart';
import 'secure_credentials_service.dart';

class GoogleDriveDownloader {
  GoogleDriveDownloader._();

  static final GoogleDriveDownloader instance = GoogleDriveDownloader._();

  static const String _prefsKey = 'gemma_model_downloaded';
  static const String _modelVersionKey = 'gemma_model_version';
  static const String _currentVersion = 'gguf_v1';
  static const String _gemmaModelName = 'gemma-2b-studyplan-q4.gguf';

  // ========== MODIFIED: Added 'Past_Paper' to the list ==========
  static const List<String> _pastPaperFolderNames = [
    'Past_Paper', // <-- your new folder name
    'PastPapers_Final',
    'PastPapers',
  ];
  static const String _gemmaFolderName = 'Gemma';

  static const String _serviceAccountEmail =
      'drive-sync-bot@axon-34b8c.iam.gserviceaccount.com';

  String? _privateKey;
  auth.AutoRefreshingAuthClient? _authClient;
  drive.DriveApi? _driveApi;

  final ValueNotifier<GemmaDownloadState> _state =
      ValueNotifier<GemmaDownloadState>(const GemmaDownloadState());

  ValueNotifier<GemmaDownloadState> get state => _state;

  final _progressController = StreamController<double>.broadcast();
  Stream<double> get progressStream => _progressController.stream;

  bool _isPaused = false;

  static const int _maxRetries = 3;
  static const int _maxConcurrentDownloads = 4;
  static const int _pageSize = 1000;

  final _mutex = _Mutex();
  final _fileIndex = _FileIndex();

  Future<void> initialize(
      {String? privateKey, bool forceDownload = false}) async {
    final credentials = SecureCredentialsService();
    await credentials.initialize();

    String? key = privateKey;
    if (key == null || key.isEmpty || key.contains('[YOUR_PRIVATE_KEY_HERE]')) {
      final allCreds = await credentials.getAllCredentials();
      key = allCreds.effectiveGoogleDriveKey;
    }

    if (key == null || key.isEmpty || !credentials.isValidApiKey(key)) {
      debugPrint(
        'Google Drive: No valid private key found. Set up in Settings > API Keys. Downloads disabled.',
      );
      return;
    }

    _privateKey = key.trim();
    debugPrint(
        'Google Drive: Private key loaded, length: ${_privateKey!.length} chars');

    // Check offline mode OR force flag for downloads
    final offlineEnabled = await _isOfflineModeEnabled();
    if ((offlineEnabled || forceDownload) && _privateKey != null) {
      debugPrint(
        'Google Drive: Download mode enabled, starting downloads...',
      );
      // Only download past papers (not Gemma by default due to size)
      unawaited(_downloadPastPapersForUser());
    }
  }

  Future<bool> _isOfflineModeEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool('offline_mode_enabled') ?? false;
  }

  Future<void> _downloadPastPapersForUser() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final subjectsList = prefs.getStringList('userSubjects');
      List<String> subjects = subjectsList ?? [];

      if (subjects.isEmpty) {
        final subjectsJson = prefs.getString('userSubjects');
        if (subjectsJson != null && subjectsJson.isNotEmpty) {
          subjects = List<String>.from(jsonDecode(subjectsJson));
        }
      }

      debugPrint('Auto-download: found subjects: $subjects');
      if (subjects.isNotEmpty) {
        await downloadPastPapersForSubjects(subjects);
      }
    } catch (e, stack) {
      debugPrint('Auto-download past papers failed: $e\n$stack');
    }
  }

  Future<bool> isModelDownloaded() async {
    final prefs = await SharedPreferences.getInstance();
    final isComplete = prefs.getBool(_prefsKey) ?? false;
    final version = prefs.getString(_modelVersionKey) ?? '';
    if (!isComplete || version != _currentVersion) return false;

    final modelDir = await _modelDirectory;
    final modelFile = File('$modelDir/$_gemmaModelName');
    if (!modelFile.existsSync()) return false;

    final size = await modelFile.length();
    return size > 100 * 1024 * 1024;
  }

  Future<String> get _modelDirectory async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${appDir.path}/models');
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return modelDir.path;
  }

  Future<void> _ensureAuthenticated() async {
    if (_driveApi != null) return;

    try {
      debugPrint('🚀 AXON_DRIVE: Starting Authentication...');

      if (_privateKey == null) {
        throw Exception('Private key not set');
      }

      final formattedKey = _privateKey!
          .replaceAll('\r\n', '\n')
          .replaceAll('\r', '\n')
          .replaceAll('\\n', '\n');

      final accountCredentials = auth.ServiceAccountCredentials(
        _serviceAccountEmail,
        auth.ClientId('', ''),
        formattedKey,
      );

      final client = await auth.clientViaServiceAccount(accountCredentials, [
        drive.DriveApi.driveReadonlyScope,
      ]).timeout(const Duration(seconds: 10));

      _authClient = client;
      _driveApi = drive.DriveApi(client);
      debugPrint('✅ AXON_DRIVE: Auth Successful');
    } catch (e, stack) {
      debugPrint('❌ AXON_DRIVE: Auth Failed: $e\n$stack');
      _driveApi = null;
      rethrow;
    }
  }

  Future<void> testConnection() async {
    try {
      debugPrint('🚀 AXON_DRIVE: Starting connection test...');
      await _ensureAuthenticated();

      if (_driveApi == null) {
        debugPrint('❌ AXON_DRIVE: Drive API not initialized');
        return;
      }

      debugPrint('✅ AXON_DRIVE: Auth successful, testing folder access...');

      final result = await _driveApi!.files.list(
        q: "name = '$_gemmaFolderName' and mimeType = 'application/vnd.google-apps.folder'",
        $fields: "files(id,name)",
      );

      if (result.files == null || result.files!.isEmpty) {
        debugPrint(
          '⚠️ AXON_DRIVE: Gemma folder NOT FOUND.',
        );
        debugPrint(
          '   Please check:',
        );
        debugPrint(
          '   1. Folder named "$_gemmaFolderName" exists in Google Drive',
        );
        debugPrint(
          '   2. Service account $_serviceAccountEmail has been invited as Viewer',
        );

        // Try to list first few folders to help debug
        final allFolders = await _driveApi!.files.list(
          q: "mimeType = 'application/vnd.google-apps.folder'",
          $fields: "files(id,name)",
          pageSize: 20,
        );
        if (allFolders.files != null && allFolders.files!.isNotEmpty) {
          debugPrint('   Available folders:');
          for (final f in allFolders.files!) {
            debugPrint('   - ${f.name}');
          }
        }
      } else {
        debugPrint(
          '📂 AXON_DRIVE: Gemma folder Found! ID: ${result.files!.first.id}',
        );
      }
    } catch (e, stack) {
      debugPrint('❌ AXON_DRIVE: Test Failed: $e\n$stack');
    }
  }

  Future<String?> _findFolder(String name) async {
    if (_driveApi == null) return null;

    try {
      debugPrint('AXON_DRIVE: Searching for folder: "$name"');

      // Try exact match first, then case-insensitive partial match
      var result = await _driveApi!.files.list(
        q: "name = '$name' and mimeType = 'application/vnd.google-apps.folder'",
        $fields: "files(id,name)",
      );

      if (result.files != null && result.files!.isNotEmpty) {
        debugPrint(
            'AXON_DRIVE: Found folder exact match: ${result.files!.first.name} (${result.files!.first.id})');
        return result.files!.first.id;
      }

      // Try case-insensitive search
      debugPrint('AXON_DRIVE: Trying fuzzy search for: "$name"');
      result = await _driveApi!.files.list(
        q: "name contains '$name' and mimeType = 'application/vnd.google-apps.folder'",
        $fields: "files(id,name)",
        pageSize: 10,
      );

      if (result.files != null && result.files!.isNotEmpty) {
        debugPrint(
            'AXON_DRIVE: Found folder (fuzzy match): ${result.files!.first.name}');
        return result.files!.first.id;
      }

      debugPrint('AXON_DRIVE: Folder not found: $name');
      return null;
    } catch (e) {
      debugPrint('Error finding folder $name: $e');
      return null;
    }
  }

  Future<List<Map<String, dynamic>>> _listFilesInFolder(String folderId) async {
    if (_driveApi == null) return [];

    final allFiles = <Map<String, dynamic>>[];
    String? pageToken;

    try {
      do {
        final result = await _driveApi!.files.list(
          q: "'$folderId' in parents and trashed = false",
          $fields: 'files(id,name,size),nextPageToken',
          pageSize: _pageSize,
          pageToken: pageToken,
        );

        if (result.files != null) {
          allFiles.addAll(
            result.files!.map(
              (f) => {
                'id': f.id!,
                'name': f.name!,
                'size': int.tryParse(f.size ?? '0') ?? 0,
              },
            ),
          );
        }
        pageToken = result.nextPageToken;
      } while (pageToken != null);
    } catch (e) {
      debugPrint('Error listing files in folder $folderId: $e');
    }

    return allFiles;
  }

  Future<List<Map<String, dynamic>>> _listFilesRecursive(
    String folderId, {
    String currentPath = '',
  }) async {
    if (_driveApi == null) return [];

    final items = await _listDriveItems(folderId);
    final files = <Map<String, dynamic>>[];

    for (final item in items) {
      final itemName = item['name'] as String? ?? '';
      final itemId = item['id'] as String? ?? '';
      final itemMimeType = item['mimeType'] as String? ?? '';
      final nextPath =
          currentPath.isEmpty ? itemName : '$currentPath/$itemName';

      if (itemMimeType == 'application/vnd.google-apps.folder') {
        files.addAll(await _listFilesRecursive(itemId, currentPath: nextPath));
        continue;
      }

      files.add({
        'id': itemId,
        'name': itemName,
        'size': item['size'] as int? ?? 0,
        'relativePath': nextPath,
      });
    }

    return files;
  }

  Future<List<Map<String, dynamic>>> _listDriveItems(String folderId) async {
    if (_driveApi == null) return [];

    final allFiles = <Map<String, dynamic>>[];
    String? pageToken;

    try {
      do {
        final result = await _driveApi!.files.list(
          q: "'$folderId' in parents and trashed = false",
          $fields: 'files(id,name,mimeType,size),nextPageToken',
          pageSize: _pageSize,
          pageToken: pageToken,
        );

        if (result.files == null) return [];

        allFiles.addAll(
          result.files!.where((f) => f.id != null && f.name != null).map(
                (f) => {
                  'id': f.id!,
                  'name': f.name!,
                  'mimeType': f.mimeType ?? '',
                  'size': int.tryParse(f.size ?? '0') ?? 0,
                },
              ),
        );

        pageToken = result.nextPageToken;
      } while (pageToken != null);
    } catch (e) {
      debugPrint('Error listing drive items in folder $folderId: $e');
    }

    return allFiles;
  }

  Future<String?> _findFirstFolder(List<String> candidates) async {
    for (final name in candidates) {
      final id = await _findFolder(name);
      if (id != null) return id;
    }
    return null;
  }

  @visibleForTesting
  static bool subjectMatchesPaperFile(String subject, String fileName) {
    final normalizedSubject = PastPaperService.normalizeSubjectName(
      subject,
    ).toLowerCase();
    final normalizedFile = fileName.toLowerCase();
    final examCodes = PastPaperService.examCodesForSubject(subject);

    if (examCodes.any(
      (code) => normalizedFile.startsWith('${code.toLowerCase()}_'),
    )) {
      return true;
    }

    final compactSubject = normalizedSubject.replaceAll(
      RegExp(r'[^a-z0-9]+'),
      '',
    );
    final compactFile = normalizedFile.replaceAll(RegExp(r'[^a-z0-9]+'), '');
    return compactSubject.isNotEmpty && compactFile.contains(compactSubject);
  }

  Future<void> _downloadFile(
    String fileId,
    String destPath, {
    void Function(double)? onProgress,
    int retryCount = 0,
  }) async {
    if (_driveApi == null) return;

    try {
      final fileMetadata = await _driveApi!.files
          .get(fileId, $fields: 'id,name,size') as drive.File;
      final int totalSize = int.tryParse(fileMetadata.size ?? '0') ?? 0;
      debugPrint('📥 Downloading: ${fileMetadata.name} ($totalSize bytes)');

      final media = await _driveApi!.files.get(
        fileId,
        downloadOptions: drive.DownloadOptions.fullMedia,
      ) as drive.Media;

      final file = File(destPath);
      final tempFile = File('$destPath.part');
      if (await tempFile.exists()) {
        await tempFile.delete();
      }
      int downloaded = 0;
      final IOSink sink = tempFile.openWrite();
      var sinkClosed = false;

      try {
        await for (final List<int> data in media.stream) {
          sink.add(data);
          downloaded += data.length;
          if (totalSize > 0 && onProgress != null) {
            onProgress(downloaded / totalSize);
          }
        }

        await sink.flush();
        await sink.close();
        sinkClosed = true;
      } finally {
        if (!sinkClosed) {
          await sink.close();
        }
      }
      if (totalSize > 0 && downloaded != totalSize) {
        throw Exception(
          'Incomplete download for ${fileMetadata.name}: $downloaded of $totalSize bytes',
        );
      }
      if (await file.exists()) {
        await file.delete();
      }
      await tempFile.rename(destPath);
      debugPrint('✨ Download Finished: $destPath');
    } catch (e) {
      // Clean up partial file
      if (File(destPath).existsSync()) {
        try {
          await File(destPath).delete();
        } catch (_) {}
      }
      final tempPath = '$destPath.part';
      if (File(tempPath).existsSync()) {
        try {
          await File(tempPath).delete();
        } catch (_) {}
      }

      if (retryCount < _maxRetries) {
        final delay = Duration(
          milliseconds:
              (100 * pow(2, retryCount) + Random().nextInt(100)).toInt(),
        );
        debugPrint('⚠️ Download attempt ${retryCount + 1} failed: $e');
        debugPrint('🔄 Retrying in ${delay.inMilliseconds}ms...');
        await Future.delayed(delay);
        await _downloadFile(
          fileId,
          destPath,
          onProgress: onProgress,
          retryCount: retryCount + 1,
        );
      } else {
        debugPrint('❌ Download failed after $_maxRetries attempts');
        rethrow;
      }
    }
  }

  Future<void> _downloadFilesConcurrent(
    List<Map<String, dynamic>> files,
    String destDir, {
    void Function(String, int, int)? onFileProgress,
  }) async {
    if (files.isEmpty) return;

    final semaphore = _Semaphore(_maxConcurrentDownloads);
    final futures = <Future>[];

    for (var i = 0; i < files.length; i++) {
      final file = files[i];
      final index = i;

      futures.add(
        semaphore.run(() async {
          try {
            final fileName = file['name'] as String;
            final lowerName = fileName.toLowerCase();

            // Skip non-PDF or non-question paper files
            if (!lowerName.endsWith('.pdf')) return;
            if (!lowerName.contains('_qp_') && !lowerName.contains('_ms_')) {
              return;
            }

            final fileId = file['id'] as String;
            final destPath = '$destDir/$fileName';
            final expectedSize = file['size'] as int? ?? 0;
            final existingFile = File(destPath);

            if (await _isUsableDownload(existingFile, expectedSize)) {
              debugPrint('Already exists: $fileName');
            } else {
              if (await existingFile.exists()) {
                await existingFile.delete();
              }
              debugPrint('Downloading: $fileName');
              await _downloadFile(fileId, destPath);
            }
            onFileProgress?.call(fileName, index + 1, files.length);
          } catch (e) {
            debugPrint('Failed to download file: $e');
          }
        }),
      );
    }

    await Future.wait(futures);
  }

  Future<bool> _isUsableDownload(File file, int expectedSize) async {
    if (!await file.exists()) return false;
    final actualSize = await file.length();
    if (actualSize <= 0) return false;
    return expectedSize <= 0 || actualSize == expectedSize;
  }

  Future<void> downloadGemmaModel({
    void Function(double progress)? onProgress,
  }) async {
    return _mutex.run(() async {
      if (_privateKey == null) {
        debugPrint(
            '❌ AXON_DRIVE: Cannot download Gemma - private key not configured');
        _state.value = const GemmaDownloadState(
          isDownloading: false,
          error:
              'Google Drive not configured. Check GOOGLE_PRIVATE_KEY in .env',
          status: 'Error: Drive not configured',
        );
        return;
      }

      debugPrint('AXON_DRIVE: Starting Gemma model download...');
      final alreadyDownloaded = await isModelDownloaded();
      if (alreadyDownloaded) {
        final modelDir = await _modelDirectory;
        final modelFile = File('$modelDir/$_gemmaModelName');
        if (await modelFile.exists()) {
          final size = await modelFile.length();
          _state.value = GemmaDownloadState(
            isComplete: true,
            progress: 1.0,
            status:
                'Ready (${(size / 1024 / 1024 / 1024).toStringAsFixed(1)}GB)',
          );
          return;
        }
      }

      _state.value = const GemmaDownloadState(
        isDownloading: true,
        progress: 0.0,
        status: 'Connecting to Google Drive...',
      );

      try {
        await _ensureAuthenticated();

        _state.value = _state.value.copyWith(
          status: 'Finding Gemma model folder...',
          progress: 0.05,
        );

        final folderId = await _findFolder(_gemmaFolderName);
        if (folderId == null) {
          throw Exception(
            'Gemma model folder not found. Check folder name and permissions.',
          );
        }

        debugPrint('Found Gemma folder: $folderId');

        _state.value = _state.value.copyWith(
          status: 'Scanning for model files...',
          progress: 0.1,
        );

        final allFiles = await _listFilesInFolder(folderId);

        debugPrint('Found ${allFiles.length} files in Gemma folder');
        for (final f in allFiles.take(5)) {
          debugPrint('  - ${f['name']} (${f['size']} bytes)');
        }

        final ggufFiles = allFiles
            .where((f) => f['name'].toString().toLowerCase().endsWith('.gguf'))
            .toList();

        if (ggufFiles.isEmpty) {
          debugPrint(
              'No .gguf files found. Available files: ${allFiles.map((f) => f['name']).join(', ')}');
          throw Exception('No .gguf model files found in Gemma folder');
        }

        final targetFile = ggufFiles.firstWhere(
          (f) => f['name'] == _gemmaModelName,
          orElse: () => ggufFiles.first,
        );

        final fileId = targetFile['id'] as String;
        final fileName = targetFile['name'] as String;
        final fileSize = targetFile['size'] as int? ?? 0;

        debugPrint('Target model: $fileName ($fileSize bytes)');

        final modelDir = await _modelDirectory;

        _state.value = _state.value.copyWith(
          status: 'Downloading $fileName',
          progress: 0.1,
        );

        final destPath = '$modelDir/$fileName';
        await _downloadFile(
          fileId,
          destPath,
          onProgress: (fileProgress) {
            final overall = 0.1 + 0.85 * fileProgress;
            onProgress?.call(overall);
            _progressController.add(overall);
          },
        );

        if (!_isPaused) {
          final modelDir = await _modelDirectory;
          final modelFile = File('$modelDir/$_gemmaModelName');
          final size = await modelFile.length();

          final prefs = await SharedPreferences.getInstance();
          await prefs.setBool(_prefsKey, true);
          await prefs.setString(_modelVersionKey, _currentVersion);

          _state.value = GemmaDownloadState(
            isComplete: true,
            progress: 1.0,
            status:
                'Ready (${(size / 1024 / 1024 / 1024).toStringAsFixed(1)}GB)',
          );
        }
      } catch (e, stack) {
        debugPrint('Gemma download error: $e\n$stack');
        _state.value = GemmaDownloadState(
          error: e.toString(),
          status: 'Error: ${e.toString()}',
        );
      }
    });
  }

  Future<void> downloadPastPapersForSubjects(
    List<String> subjects, {
    void Function(String subject, int current, int total)? onProgress,
  }) async {
    return _mutex.run(() async {
      if (_privateKey == null) return;

      debugPrint('=== Starting past papers download ===');
      debugPrint('Subjects: $subjects');
      debugPrint('Private key available: ${_privateKey != null}');

      try {
        await _ensureAuthenticated();
        debugPrint('Authenticated successfully');

        final folderId = await _findFirstFolder(_pastPaperFolderNames);
        debugPrint('Past papers folder ID: $folderId');
        if (folderId == null) {
          debugPrint(
            'Past papers folder not found in Google Drive. Checked: ${_pastPaperFolderNames.join(', ')}',
          );
          return;
        }

        final appDir = await getApplicationDocumentsDirectory();
        final pdfDir = Directory('${appDir.path}/past_papers');
        if (!await pdfDir.exists()) {
          await pdfDir.create(recursive: true);
        }

        final allFiles = await _listFilesRecursive(folderId);
        debugPrint('Total files found: ${allFiles.length}');

        // Build file index once
        _fileIndex.build(allFiles);

        int subjectIndex = 0;
        final manifest = <String, List<String>>{};

        for (final subject in subjects) {
          if (_isPaused) break;

          final normalizedSubject = PastPaperService.normalizeSubjectName(
            subject,
          );
          final subjectDir = Directory('${pdfDir.path}/$normalizedSubject');
          if (!await subjectDir.exists()) {
            await subjectDir.create(recursive: true);
          }

          // Use index for fast lookup
          final codes = PastPaperService.examCodesForSubject(subject);
          final matchingFiles = _fileIndex.findBySubject(
            subject,
            codes.toSet(),
          );

          debugPrint(
            'Subject "$normalizedSubject": ${matchingFiles.length} matching files',
          );
          onProgress?.call(subject, subjectIndex + 1, subjects.length);
          subjectIndex++;

          final downloadedForSubject = <String>[];

          // Download with concurrency
          await _downloadFilesConcurrent(
            matchingFiles,
            subjectDir.path,
            onFileProgress: (name, current, total) {
              debugPrint('[$subject] $current/$total: $name');
            },
          );

          // Collect downloaded files
          for (final file in matchingFiles) {
            final fileName = file['name'] as String;
            final destPath = '${subjectDir.path}/$fileName';
            if (File(destPath).existsSync()) {
              downloadedForSubject.add(fileName);
            }
          }

          manifest[normalizedSubject] = downloadedForSubject;
        }
        await PastPaperService().saveDownloadManifest(manifest);
        debugPrint('=== Past papers download complete ===');
      } catch (e, stack) {
        debugPrint('Error downloading past papers: $e\n$stack');
        _state.value = GemmaDownloadState(
          error: e.toString(),
          status: 'Error: ${e.toString()}',
        );
      }
    });
  }

  void pauseDownload() {
    _isPaused = true;
  }

  void resumeDownload() {
    _isPaused = false;
  }

  void cancelDownload() {
    _isPaused = false;
    _authClient?.close();
    _authClient = null;
    _driveApi = null;
    _state.value = const GemmaDownloadState();
  }

  void triggerDownload() {
    downloadGemmaModel();
  }

  void dispose() {
    _progressController.close();
    _authClient?.close();
  }
}

// Thread-safe semaphore for concurrent downloads
class _Semaphore {
  final int maxCount;
  int _currentCount = 0;
  final _queue = <Completer<void>>[];

  _Semaphore(this.maxCount);

  Future<T> run<T>(Future<T> Function() fn) async {
    while (_currentCount >= maxCount) {
      final completer = Completer<void>();
      _queue.add(completer);
      await completer.future;
    }
    _currentCount++;
    try {
      return await fn();
    } finally {
      _currentCount--;
      if (_queue.isNotEmpty) {
        _queue.removeAt(0).complete();
      }
    }
  }
}

// Mutex for thread-safe operations
class _Mutex {
  bool _locked = false;
  final _queue = <Completer<void>>[];

  Future<T> run<T>(Future<T> Function() fn) async {
    while (_locked) {
      final completer = Completer<void>();
      _queue.add(completer);
      await completer.future;
    }
    _locked = true;
    try {
      return await fn();
    } finally {
      _locked = false;
      if (_queue.isNotEmpty) {
        _queue.removeAt(0).complete();
      }
    }
  }
}

// Efficient file index for fast subject lookups
class _FileIndex {
  final Map<String, List<Map<String, dynamic>>> _byPrefix = {};
  final List<Map<String, dynamic>> _bySubject = [];

  void build(List<Map<String, dynamic>> files) {
    _byPrefix.clear();
    _bySubject.clear();

    for (final file in files) {
      final name = file['name'].toString().toLowerCase();
      _bySubject.add(file);

      // ========== FIXED REGEX ==========
      // Captures any alphanumeric prefix before the first underscore
      // Works for both numeric (0410) and alphanumeric (4MA1) exam codes
      final prefixMatch = RegExp(
        r'^([A-Za-z0-9]+)_',
        caseSensitive: false,
      ).firstMatch(name);
      if (prefixMatch != null) {
        final prefix = prefixMatch.group(1)!.toUpperCase();
        (_byPrefix[prefix] ??= []).add(file);
      }
    }
  }

  List<Map<String, dynamic>> findBySubject(String subject, Set<String> codes) {
    final result = <Map<String, dynamic>>{};

    for (final code in codes) {
      result.addAll(_byPrefix[code] ?? {});
    }

    final compact = PastPaperService.normalizeSubjectName(subject)
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]'), '');
    if (compact.isNotEmpty) {
      for (final file in _bySubject) {
        final fileCompact = file['name'].toString().toLowerCase().replaceAll(
              RegExp(r'[^a-z0-9]'),
              '',
            );
        if (fileCompact.contains(compact)) {
          result.add(file);
        }
      }
    }

    return result.toList();
  }
}

class GemmaDownloadState {
  final bool isDownloading;
  final double progress;
  final String status;
  final bool isComplete;
  final String? error;

  const GemmaDownloadState({
    this.isDownloading = false,
    this.progress = 0.0,
    this.status = '',
    this.isComplete = false,
    this.error,
  });

  GemmaDownloadState copyWith({
    bool? isDownloading,
    double? progress,
    String? status,
    bool? isComplete,
    String? error,
  }) {
    return GemmaDownloadState(
      isDownloading: isDownloading ?? this.isDownloading,
      progress: progress ?? this.progress,
      status: status ?? this.status,
      isComplete: isComplete ?? this.isComplete,
      error: error,
    );
  }
}
