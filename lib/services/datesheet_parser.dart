import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:crypto/crypto.dart';
import 'package:path_provider/path_provider.dart';

import 'exam_data_service.dart';
import 'glm_ocr_service.dart';

class DatesheetParser {
  DatesheetParser._();
  static final DatesheetParser instance = DatesheetParser._();

  static const String _cacheDirName = 'datesheet_cache';
  final Set<String> _processedHashes = {};

  Future<String> get _cacheDirectory async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.parent.path}/$_cacheDirName');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir.path;
  }

  Future<bool> processDatesheetPdf(File pdfFile, String sourceUrl) async {
    final hash = md5.convert(await pdfFile.readAsBytes()).toString();
    if (_processedHashes.contains(hash)) {
      debugPrint('Datesheet already processed: $hash');
      return false;
    }

    try {
      final events = await _parsePdf(pdfFile, sourceUrl);
      if (events.isEmpty) {
        debugPrint('No exam events extracted from: ${pdfFile.path}');
        return false;
      }

      final cacheDir = await _cacheDirectory;
      final fileName = _generateFileName(sourceUrl);
      final outputFile = File('$cacheDir/$fileName');

      final data = {
        'source': sourceUrl,
        'hash': hash,
        'processed_at': DateTime.now().toIso8601String(),
        'events': events.map((e) => e.toJson()).toList(),
      };

      await outputFile.writeAsString(jsonEncode(data));
      _processedHashes.add(hash);

      debugPrint('Cached ${events.length} exam events to: $fileName');
      return true;
    } catch (e) {
      debugPrint('Failed to process datesheet: $e');
      return false;
    }
  }

  Future<List<ExamEvent>> _parsePdf(File pdfFile, String sourceUrl) async {
    final events = <ExamEvent>[];
    final content = await _extractTextFromPdf(pdfFile);

    if (content.isEmpty) {
      debugPrint('Empty content extracted from PDF');
      return events;
    }

    // Parse using multiple strategies based on content format
    events.addAll(_parseSyllabusViewFormat(content, sourceUrl));
    events.addAll(_parseDetailedFormat(content, sourceUrl));

    // Deduplicate by (subject, component, date)
    final seen = <String>{};
    final uniqueEvents = <ExamEvent>[];
    for (final event in events) {
      final key =
          '${event.subject}_${event.component}_${event.date.toIso8601String()}';
      if (!seen.contains(key)) {
        seen.add(key);
        uniqueEvents.add(event);
      }
    }

    return uniqueEvents;
  }

  List<ExamEvent> _parseSyllabusViewFormat(String content, String sourceUrl) {
    final events = <ExamEvent>[];
    final lines = content.split('\n');

    String? board = _detectBoard(sourceUrl);

    // Pattern: Subject name, code, duration, date (syllabus view format)
    // Example: "English (Primary) 0058/01 1 hour Monday 09 March 2026"
    final syllabusPattern = RegExp(
      r'^([A-Za-z\s]+(?:Primary|Lower Secondary)?)\s+(\d{4}/\d{2})\s+(.+?)\s+((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
      caseSensitive: false,
    );

    for (final line in lines) {
      final match = syllabusPattern.firstMatch(line.trim());
      if (match != null) {
        final subject = match.group(1)?.trim() ?? '';
        final component = match.group(2) ?? '';
        final duration = match.group(3) ?? '';
        final dateStr = match.group(4) ?? '';

        final parsedDate = _parseDate(dateStr);
        if (parsedDate != null && subject.isNotEmpty) {
          final times = _inferTimesFromDuration(duration);
          events.add(ExamEvent(
            board: board ?? 'Cambridge',
            subject: subject,
            component: component,
            date: parsedDate,
            startTime: times['start'] ?? '08:00',
            endTime: times['end'] ?? '09:00',
          ));
        }
      }
    }

    return events;
  }

  List<ExamEvent> _parseDetailedFormat(String content, String sourceUrl) {
    final events = <ExamEvent>[];
    final lines = content.split('\n');

    String? board = _detectBoard(sourceUrl);
    String currentSubject = '';
    String currentComponent = '';
    DateTime? currentDate;

    for (final rawLine in lines) {
      final line = rawLine.trim();
      if (line.isEmpty) continue;

      // Detect board from header
      board ??= _detectBoard(line);

      // Detect subject line (e.g., "MATHEMATICS (9709)", "PHYSICS (9702)")
      final subjectMatch = RegExp(
        r'^([A-Z\s&]+)\s*\((\d{4})\)',
      ).firstMatch(line);

      if (subjectMatch != null) {
        currentSubject = subjectMatch.group(1)!.trim();
        currentComponent = subjectMatch.group(2)!;
        continue;
      }

      // Detect date line (e.g., "15 May 2026", "2026-05-15")
      final dateMatch = _parseDate(line);
      if (dateMatch != null) {
        currentDate = dateMatch;
        continue;
      }

      // Detect exam entry (e.g., "Paper 1", "Paper 2", "Paper 31")
      if (currentDate != null && currentSubject.isNotEmpty) {
        final paperMatch = RegExp(
          r'^(Paper\s+\d+[A-Z]?)\s*(.*)',
          caseSensitive: false,
        ).firstMatch(line);

        if (paperMatch != null) {
          final paper = paperMatch.group(1)!.trim();
          final timeInfo = paperMatch.group(2)?.trim() ?? '';
          final times = _extractTimes(timeInfo);

          events.add(ExamEvent(
            board: board ?? 'Cambridge',
            subject: currentSubject,
            component: '$currentComponent $paper',
            date: currentDate,
            startTime: times['start'] ?? '09:00',
            endTime: times['end'] ?? '12:00',
          ));
        }
      }
    }

    return events;
  }

  Map<String, String> _inferTimesFromDuration(String duration) {
    // Extract numeric duration and infer start/end times
    // Common: "1 hour", "45 minutes", "1 hour 10 minutes"
    final hourMatch =
        RegExp(r'(\d+)\s*hour').firstMatch(duration.toLowerCase());
    final minMatch =
        RegExp(r'(\d+)\s*minute').firstMatch(duration.toLowerCase());

    int durationMinutes = 0;
    if (hourMatch != null) {
      durationMinutes += (int.tryParse(hourMatch.group(1)!) ?? 0) * 60;
    }
    if (minMatch != null) {
      durationMinutes += int.tryParse(minMatch.group(1)!) ?? 0;
    }

    // Default to morning session (08:00) if not specified
    if (durationMinutes == 0) {
      durationMinutes = 60; // Default 1 hour
    }

    final startHour = 8; // Default morning start
    final endHour = startHour + (durationMinutes ~/ 60);
    final endMinute = durationMinutes % 60;

    return {
      'start':
          '$startHour:${durationMinutes % 60 == 0 ? '00' : (durationMinutes % 60).toString().padLeft(2, '0')}',
      'end':
          '$endHour:${endMinute == 0 ? '00' : endMinute.toString().padLeft(2, '0')}',
    };
  }

  Future<String> _extractTextFromPdf(File pdfFile) async {
    // Method 1: Use GLMOcrService (wraps Python GLM OCR + fallback)
    try {
      final glService = GLMOcrService.instance;
      final text = await glService.extractTextFromPdf(pdfFile.path);
      if (text.isNotEmpty) {
        debugPrint(
            'GLM OCR extraction successful for ${pdfFile.path.split('/').last}');
        return text;
      }
    } catch (e) {
      debugPrint('GLM OCR service failed: $e');
    }

    // Method 2: Try Python script directly
    try {
      final safePath = pdfFile.path;
      if (RegExp(r'[;\'"|`$]').hasMatch(safePath)) {
        debugPrint('Python GLM OCR skipped: unsafe characters in path');
        return '';
      }
      final result = await Process.run(
        'python',
        [
          'rerun_datesheets_glm_ocr.py',
          '--single',
          safePath,
        ],
        workingDirectory: Directory.current.path,
        runInShell: Platform.isWindows,
      );
      if (result.exitCode == 0 && result.stdout.toString().isNotEmpty) {
        debugPrint(
            'Python GLM OCR extraction successful for ${safePath.split('/').last}');
        return result.stdout.toString();
      }
    } catch (e) {
      debugPrint('Python GLM OCR extraction failed: $e');
    }

    // Method 3: Try using pdftotext command if available
    try {
      final safePath = pdfFile.path;
      if (RegExp(r'[;\'"|`$]').hasMatch(safePath)) {
        debugPrint('pdftotext skipped: unsafe characters in path');
        return '';
      }
      final result =
          await Process.run('pdftotext', [safePath, '-'], runInShell: Platform.isWindows);
      if (result.exitCode == 0 && result.stdout.toString().isNotEmpty) {
        return result.stdout.toString();
      }
    } catch (e) {
      debugPrint('pdftotext failed: $e');
    }

    debugPrint('All PDF extraction methods failed for ${pdfFile.path}');
    return '';
  }

  String? _detectBoard(String text) {
    final lower = text.toLowerCase();
    if (lower.contains('cambridge') ||
        lower.contains('caie') ||
        lower.contains('cie')) {
      return 'Cambridge';
    }
    return null;
  }

  DateTime? _parseDate(String line) {
    // Try common date formats
    final patterns = [
      RegExp(
          r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'),
      RegExp(r'(\d{4})-(\d{2})-(\d{2})'),
      RegExp(r'(\d{1,2})/(\d{1,2})/(\d{4})'),
      // Day of week + date format: "Monday 09 March 2026"
      RegExp(
          r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
          caseSensitive: false),
    ];

    for (final pattern in patterns) {
      final match = pattern.firstMatch(line);
      if (match != null) {
        try {
          // Try standard ISO format first
          if (RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(line.trim())) {
            return DateTime.parse(line.trim());
          }

          // Manual parsing for other formats
          if (match.groupCount >= 3) {
            int day, month, year;
            // Handle "Monday 09 March 2026" format
            if (match.groupCount == 4 && match.group(2) != null) {
              day = int.tryParse(match.group(2)!) ?? 0;
              month = _monthToInt(match.group(3)!);
              year = int.tryParse(match.group(4)!) ?? 0;
            } else {
              day = int.tryParse(match.group(1)!) ?? 0;
              month = _monthToInt(match.group(2)!);
              year = int.tryParse(match.group(3)!) ?? 0;
            }

            if (day >= 1 && day <= 31 && month >= 1 && month <= 12) {
              return DateTime(year, month, day);
            }
          }
        } catch (_) {
          continue;
        }
      }
    }
    return null;
  }

  int _monthToInt(String month) {
    const months = {
      'January': 1,
      'February': 2,
      'March': 3,
      'April': 4,
      'May': 5,
      'June': 6,
      'July': 7,
      'August': 8,
      'September': 9,
      'October': 10,
      'November': 11,
      'December': 12,
    };
    return months[month] ?? 1;
  }

  Map<String, String> _extractTimes(String text) {
    final times = <String, String>{};
    final timePattern = RegExp(r'(\d{1,2}:\d{2})\s*[-–—]\s*(\d{1,2}:\d{2})');
    final match = timePattern.firstMatch(text);
    if (match != null) {
      times['start'] = match.group(1)!;
      times['end'] = match.group(2)!;
    }
    return times;
  }

  String _generateFileName(String sourceUrl) {
    final hash = md5.convert(utf8.encode(sourceUrl)).toString();
    final now = DateTime.now();
    return 'datesheet_${now.year}_${now.month}_${now.day}_$hash.json';
  }

  Future<void> clearCache() async {
    final cacheDir = await _cacheDirectory;
    final dir = Directory(cacheDir);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
    _processedHashes.clear();
  }
}
