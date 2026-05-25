import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'app_state.dart';
import 'question_intelligence_service.dart';
import 'study_catalog.dart';
import 'resource_crawler_service.dart';

class AskAxonContextService {
  AskAxonContextService._();

  static final AskAxonContextService instance = AskAxonContextService._();

  Future<String> buildContext({
    MetricsState? metrics,
    TimerState? timerState,
    String? currentSubject,
    String? currentChapter,
    String? activePaperTitle,
    String? activePaperPath,
    String? extraContext,
    int recentMistakeLimit = 3,
    bool includeStudyCatalog = true,
    bool includeSyllabus = true,
    bool includeResources = true,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final userBoard = prefs.getString('userBoard') ?? '';
    final userSubjects = prefs.getStringList('userSubjects') ?? const [];
    final recentMistakes =
        await QuestionIntelligenceService.instance.loadMistakes();

    final activeTimer = timerState ?? await _loadPersistedTimer();
    final resolvedSubject = _pickFirstNonEmpty([
      currentSubject,
      activeTimer?.subject,
      metrics?.primarySubject,
      recentMistakes.isNotEmpty ? recentMistakes.first.subject : null,
    ]);
    final resolvedChapter = _pickFirstNonEmpty([
      currentChapter,
      activeTimer?.chapter,
      recentMistakes.isNotEmpty ? recentMistakes.first.chapter : null,
    ]);

    final buffer = StringBuffer();

    // Build comprehensive study resource context
    if (includeStudyCatalog) {
      buffer.writeln('=== STUDY CATALOG ===');
      buffer.writeln(await _buildStudyCatalogContext());
      buffer.writeln();
    }

    if (includeSyllabus && resolvedSubject.isNotEmpty) {
      buffer.writeln('=== SYLLABUS CONTENT ===');
      buffer.writeln(await _buildSyllabusContext(resolvedSubject));
      buffer.writeln();
    }

    if (includeResources) {
      buffer.writeln('=== LEARNING RESOURCES ===');
      if (resolvedSubject.isNotEmpty && resolvedChapter.isNotEmpty) {
        buffer.writeln(
            await _buildResourceContext(resolvedSubject, resolvedChapter));
      }
      buffer.writeln();
    }

    // User info
    if (userBoard.isNotEmpty) {
      buffer.writeln('=== USER PROFILE ===');
      buffer.writeln('Board: $userBoard');
    }
    if (userSubjects.isNotEmpty) {
      buffer.writeln('User subjects: ${userSubjects.join(', ')}');
    }
    if (metrics != null) {
      buffer.writeln('Primary subject: ${metrics.primarySubject}');
      buffer.writeln(
          'Study hours today: ${metrics.activeStudyHours.toStringAsFixed(1)} / ${metrics.targetStudyHours.toStringAsFixed(1)}');
      buffer.writeln(
          'Performance score: ${(metrics.predictedPerformance * 100).round()}%');
      buffer.writeln(
          'Sleep: ${metrics.sleepHours.toStringAsFixed(1)}h, Screen time: ${metrics.screenTimeHours.toStringAsFixed(1)}h');
    }
    if (resolvedSubject.isNotEmpty) {
      buffer.writeln('Current subject: $resolvedSubject');
    }
    if (resolvedChapter.isNotEmpty) {
      buffer.writeln('Current chapter: $resolvedChapter');
    }
    if (activeTimer != null && activeTimer.subject.isNotEmpty) {
      buffer.writeln(
          'Active timer: ${activeTimer.subject} · ${activeTimer.chapter} · ${activeTimer.elapsed.inMinutes} min elapsed · ${activeTimer.isRunning ? 'running' : 'paused'}');
      if (activeTimer.templateName.isNotEmpty) {
        buffer.writeln('Session template: ${activeTimer.templateName}');
      }
    }
    if ((activePaperTitle ?? '').trim().isNotEmpty) {
      buffer.writeln('Active paper: ${activePaperTitle!.trim()}');
    }
    if ((activePaperPath ?? '').trim().isNotEmpty) {
      buffer.writeln('Paper source: ${activePaperPath!.trim()}');
    }

    final trimmedMistakes = recentMistakes.take(recentMistakeLimit).toList();
    if (trimmedMistakes.isNotEmpty) {
      buffer.writeln('Recent mistakes:');
      for (final item in trimmedMistakes) {
        final parts = [
          if (item.subject.isNotEmpty) item.subject,
          if (item.chapter.isNotEmpty) item.chapter,
          if (item.topic.isNotEmpty) item.topic,
        ].join(' · ');
        buffer.writeln(
          '- Q${item.questionNumber}: ${parts.isEmpty ? item.sourceTitle : parts} | ${_compact(item.questionText, 140)}',
        );
      }
    }

    if ((extraContext ?? '').trim().isNotEmpty) {
      buffer.writeln('Attached context:');
      buffer.writeln(extraContext!.trim());
    }
    return buffer.toString().trim();
  }

  Future<String> _buildStudyCatalogContext() async {
    try {
      final catalog = StudyCatalog();
      final data = await catalog.load();

      if (data.isEmpty) return 'No study catalog data available';

      final buffer = StringBuffer();

      for (final entry in data.entries) {
        final subject = entry.key;
        final List<String> chapters = entry.value;

        buffer.writeln('Subject: $subject');
        buffer.writeln('  Chapters: ${chapters.join(', ')}');
      }

      return buffer.toString().trim();
    } catch (e) {
      return 'Error loading study catalog: $e';
    }
  }

  Future<String> _buildSyllabusContext(String subject) async {
    try {
      final catalog = StudyCatalog();
      final data = await catalog.load();
      final prefs = await SharedPreferences.getInstance();
      final board = prefs.getString('userBoard') ?? 'IGCSE';

      // Get chapters for this subject and board
      final Map<String, Map<String, List<String>>> resources =
          await catalog.loadScrapedResources();
      final subjectResources = resources[subject] ?? {};
      final boardChapters = subjectResources[board] ?? [];

      final buffer = StringBuffer();
      buffer.writeln('$board $subject Syllabus:');

      if (boardChapters.isNotEmpty) {
        for (int i = 0; i < boardChapters.length; i++) {
          buffer.writeln('  ${i + 1}. ${boardChapters[i]}');
        }
      } else {
        // Fall back to basic chapters
        final chapters = data[subject] ?? [];
        if (chapters.isNotEmpty) {
          for (int i = 0; i < chapters.length; i++) {
            buffer.writeln('  ${i + 1}. ${chapters[i]}');
          }
        }
      }

      return buffer.toString().trim();
    } catch (e) {
      return 'Error loading syllabus: $e';
    }
  }

  Future<String> _buildResourceContext(String subject, String chapter) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final uid = prefs.getString('userUid') ?? '';
      final board = prefs.getString('userBoard') ?? '';

      if (uid.isEmpty || board.isEmpty) return 'No resources available';

      final crawler = ResourceCrawlerService();
      final resources = await crawler.loadResources(
        board: board,
        subject: subject,
        chapter: chapter,
        uid: uid,
      );

      if (resources.isEmpty) {
        return 'No resources found for $subject - $chapter';
      }

      final buffer = StringBuffer();
      buffer.writeln('Resources for $subject - $chapter:');

      // Group by type
      final videos = resources.where((r) => r.type == 'video').toList();
      final notes = resources.where((r) => r.type == 'notes').toList();
      final pastPapers = resources.where((r) => r.type == 'pastPaper').toList();
      final textbooks = resources.where((r) => r.type == 'textbook').toList();

      if (videos.isNotEmpty) {
        buffer.writeln('Videos:');
        for (final r in videos.take(5)) {
          buffer.writeln('  - ${r.title}: ${r.url}');
        }
      }

      if (notes.isNotEmpty) {
        buffer.writeln('Notes:');
        for (final r in notes.take(5)) {
          buffer.writeln('  - ${r.title}: ${r.url}');
        }
      }

      if (pastPapers.isNotEmpty) {
        buffer.writeln('Past Papers:');
        for (final r in pastPapers.take(5)) {
          buffer.writeln('  - ${r.title}: ${r.url}');
        }
      }

      if (textbooks.isNotEmpty) {
        buffer.writeln('Textbook References:');
        for (final r in textbooks.take(5)) {
          buffer.writeln('  - ${r.title}: ${r.url}');
        }
      }

      return buffer.toString().trim();
    } catch (e) {
      return 'Error loading resources: $e';
    }
  }

  Future<TimerState?> _loadPersistedTimer() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('active_timer_state');
    if (raw == null || raw.isEmpty) return null;
    try {
      final map = Map<String, dynamic>.from(jsonDecode(raw));
      return TimerState(
        isRunning: map['isRunning'] == true,
        elapsed:
            Duration(seconds: (map['elapsedSeconds'] as num?)?.toInt() ?? 0),
        breakCount: (map['breakCount'] as num?)?.toInt() ?? 0,
        subject: (map['subject'] ?? '').toString(),
        chapter: (map['chapter'] ?? '').toString(),
        intensityIndex: (map['intensityIndex'] as num?)?.toDouble() ?? 0.0,
        pings: (map['pings'] as num?)?.toInt() ?? 0,
        startedAt: DateTime.tryParse((map['startedAt'] ?? '').toString()),
        isRestoring: false,
        templateId: (map['templateId'] ?? '').toString(),
        templateName: (map['templateName'] ?? '').toString(),
        targetDuration: (map['targetDurationSeconds'] as num?) == null
            ? null
            : Duration(
                seconds: (map['targetDurationSeconds'] as num).toInt(),
              ),
      );
    } catch (_) {
      return null;
    }
  }

  String _pickFirstNonEmpty(List<String?> candidates) {
    for (final candidate in candidates) {
      if ((candidate ?? '').trim().isNotEmpty) {
        return candidate!.trim();
      }
    }
    return '';
  }

  String _compact(String text, int maxChars) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.length <= maxChars) return normalized;
    return '${normalized.substring(0, maxChars)}...';
  }
}
