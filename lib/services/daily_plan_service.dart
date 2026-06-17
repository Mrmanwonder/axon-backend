import 'dart:convert';
import 'dart:math' as math show Random;
import 'package:crypto/crypto.dart';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../models/daily_plan_task.dart';
import 'backend_config.dart';
import 'firestore_service.dart';
import 'exam_repository.dart';

class DailyPlanException implements Exception {
  final String message;

  const DailyPlanException(this.message);

  @override
  String toString() => message;
}

class DailyPlanService {
  DailyPlanService({http.Client? client}) : _client = client ?? http.Client();

  static const String _backendUrl = BackendConfig.baseUrl;
  static const Duration _requestTimeout = Duration(seconds: 120);
  /// Short timeout for v2 endpoint (no Firestore — should respond in <5s)
  static const Duration _v2Timeout = Duration(seconds: 30);
  static final math.Random _random = math.Random();
  static final Map<String, Future<List<DailyPlanTask>>> _inFlightGenerations =
      {};

  final http.Client _client;
  bool _isGenerating = false;

  CollectionReference<Map<String, dynamic>> _dailyPlanCollection(String uid) =>
      AxonPaths.privateUserCollection(uid, 'daily_plan');

  Stream<List<DailyPlanTask>> watchTodayPlan(String uid, {DateTime? now}) {
    final today = _todayString(now);
    return _dailyPlanCollection(uid)
        .where('date', isEqualTo: today)
        .snapshots()
        .map(
          (snapshot) => _sortTasks(snapshot.docs
              .map((doc) => DailyPlanTask.fromJson(doc.id, doc.data()))
              .toList()),
        );
  }

  Future<List<DailyPlanTask>> getTasksForDate(String uid, String date) async {
    try {
      final snapshot =
          await _dailyPlanCollection(uid).where('date', isEqualTo: date).get();
      return _sortTasks(snapshot.docs
          .map((doc) => DailyPlanTask.fromJson(doc.id, doc.data()))
          .toList());
    } catch (e) {
      debugPrint('DailyPlanService: failed to load tasks for $date - $e');
      return const <DailyPlanTask>[];
    }
  }

  Future<List<DailyPlanTask>> getTasksForDateRange(
      String uid, String startDate, String endDate) async {
    try {
      final snapshot = await _dailyPlanCollection(uid)
          .where('date', isGreaterThanOrEqualTo: startDate)
          .where('date', isLessThanOrEqualTo: endDate)
          .get();
      return _sortTasks(snapshot.docs
          .map((doc) => DailyPlanTask.fromJson(doc.id, doc.data()))
          .toList());
    } catch (e) {
      debugPrint(
          'DailyPlanService: indexed range query failed, using local filter - $e');
      final snapshot = await _dailyPlanCollection(uid).get();
      return _sortTasks(snapshot.docs
          .map((doc) => DailyPlanTask.fromJson(doc.id, doc.data()))
          .where((task) => task.date.compareTo(startDate) >= 0)
          .where((task) => task.date.compareTo(endDate) <= 0)
          .toList());
    }
  }

  Future<void> generateTodayPlan({String? focusAreas}) async {
    if (_isGenerating) {
      debugPrint('DailyPlanService: already generating, skipping');
      return;
    }
    _isGenerating = true;
    try {
      // ── Strategy: Try v2 (fast, no-Firestore) first, then smart local ──
      final tasks = await _generateV2Plan(focusAreas: focusAreas);
      if (tasks.isEmpty) {
        debugPrint(
            'DailyPlanService: v2 returned no tasks, using smart local fallback');
        await _generateSmartLocalPlan(focusAreas: focusAreas);
      } else {
        debugPrint('DailyPlanService: v2 generated ${tasks.length} tasks');
        // Save to offline cache
        final uid = FirebaseAuth.instance.currentUser?.uid ?? '';
        final today = _todayString();
        if (uid.isNotEmpty) {
          await _saveTasksToOffline(uid, today, tasks);
        }
      }
    } catch (e) {
      debugPrint(
          'DailyPlanService: v2 generation failed, using smart local fallback — $e');
      await _generateSmartLocalPlan(focusAreas: focusAreas);
    } finally {
      _isGenerating = false;
    }
  }

  /// V2 endpoint call — no Firestore dependency, responds in <5s
  Future<List<DailyPlanTask>> _generateV2Plan({String? focusAreas}) async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final uid = user?.uid ?? '';
      final subjects = await _getUserSubjects();
      if (subjects.isEmpty) return [];

      // Get exam dates from prefs
      final prefs = await SharedPreferences.getInstance();
      final examDatesJson = prefs.getString('userExamDates') ?? '{}';
      final examDates = jsonDecode(examDatesJson) as Map<String, dynamic>;
      double targetHours = prefs.getDouble('userTargetHours') ?? 4.0;
      if (targetHours <= 0) targetHours = 4.0;

      final response = await _client
          .post(
            Uri.parse('$_backendUrl/api/v2/daily-plan/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'user_id': uid,
              'subjects': subjects,
              'target_hours': targetHours,
              'client_date': _todayString(),
              if (focusAreas != null && focusAreas.isNotEmpty)
                'focus_areas': focusAreas,
              'force': true,
              'exam_dates': examDates,
            }),
          )
          .timeout(_v2Timeout);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        debugPrint('DailyPlanService: v2 returned ${response.statusCode}: ${response.body}');
        return [];
      }

      return _tasksFromResponse(response);
    } catch (e) {
      debugPrint('DailyPlanService: _generateV2Plan failed - $e');
      return [];
    }
  }

  /// Smart local fallback — mirrors backend planner_service.py algorithm
  /// Phase-adaptive task mix, cognitive load management, FSRS-lite urgency
  Future<void> _generateSmartLocalPlan({
    String? focusAreas,
    String? uidOverride,
  }) async {
    debugPrint('DailyPlanService: _generateSmartLocalPlan() called');
    try {
      final user = FirebaseAuth.instance.currentUser;
      debugPrint('DailyPlanService: currentUser=${user?.uid}');
      final uid = uidOverride ?? user?.uid;
      if (uid == null || uid.isEmpty) {
        debugPrint('DailyPlanService: uid is null/empty');
        return;
      }
      final today = _todayString();
      debugPrint('DailyPlanService: uid=$uid, today=$today');

      final existing = await getTasksForDate(uid, today);
      debugPrint('DailyPlanService: existing tasks=${existing.length}');
      if (existing.isNotEmpty) return;

      final subjects = await _getUserSubjects();
      debugPrint('DailyPlanService: subjects=$subjects');
      if (subjects.isEmpty) {
        debugPrint('DailyPlanService: subjects is empty');
        return;
      }

      final prefs = await SharedPreferences.getInstance();
      double targetHours = prefs.getDouble('userTargetHours') ?? 4.0;
      if (targetHours <= 0) targetHours = 4.0;

      // ── Phase detection (same as backend) ──
      final phase = await _calculatePhase(subjects);
      final phaseKey = phase.name; // foundation, t30Completion, t14DeepDive, t7MockSprint
      String phaseLabel = phase.label;
      int daysToExam = await _getDaysToNearestExam(subjects);

      // ── Phase-adaptive task mix (from planner_service.py) ──
      final List<MapEntry<String, double>> phaseMix = switch (phaseKey) {
        't7MockSprint' => [
          MapEntry('mock_exam', 0.60), MapEntry('examiner_report', 0.15),
          MapEntry('command_word_drill', 0.15), MapEntry('flashcards', 0.10),
        ],
        't14DeepDive' => [
          MapEntry('past_paper', 0.40), MapEntry('command_word_drill', 0.20),
          MapEntry('review', 0.20), MapEntry('examiner_report', 0.10),
          MapEntry('flashcards', 0.10),
        ],
        't30Completion' => [
          MapEntry('practice', 0.35), MapEntry('past_paper', 0.20),
          MapEntry('deep_work', 0.20), MapEntry('review', 0.15),
          MapEntry('flashcards', 0.10),
        ],
        _ => [
          MapEntry('deep_work', 0.55), MapEntry('practice', 0.20),
          MapEntry('review', 0.15), MapEntry('flashcards', 0.10),
        ],
      };

      // ── Cognitive Load Units (from backend) ──
      const cluMap = <String, double>{
        'mock_exam': 10.0, 'past_paper': 7.0, 'deep_work': 6.0,
        'command_word_drill': 5.0, 'practice': 5.0,
        'examiner_report': 3.0, 'review': 3.0, 'flashcards': 2.0,
      };
      const dailyCluBudget = 32.0;
      const maxTasksPerSubject = 3;

      // ── Task durations (minutes) ──
      const durMap = <String, int>{
        'mock_exam': 120, 'past_paper': 60, 'deep_work': 50,
        'practice': 40, 'command_word_drill': 30,
        'review': 30, 'examiner_report': 25, 'flashcards': 20,
      };

      // ── Time windows (same proportions as backend) ──
      final totalMinutes = (targetHours * 60).toInt();
      final now = DateTime.now();
      int startHour = now.hour + 1;
      if (startHour < 8) startHour = 8;
      if (startHour > 18) startHour = 18; // cap at 6pm for scheduling

      final todayDate = DateTime.parse(today);
      final slots = <_TimeSlot>[];
      var cursor = DateTime(todayDate.year, todayDate.month, todayDate.day, startHour);

      // Peak Focus Morning (40%)
      int peakDur = (totalMinutes * 0.40).round();
      slots.add(_TimeSlot(
        window: ScheduledWindow.peakFocusMorning,
        start: cursor, end: cursor.add(Duration(minutes: peakDur)),
        cluRemaining: dailyCluBudget * 0.40,
      ));
      cursor = cursor.add(Duration(minutes: peakDur));

      // Structured Morning (20%)
      int structDur = (totalMinutes * 0.20).round();
      slots.add(_TimeSlot(
        window: ScheduledWindow.morning,
        start: cursor, end: cursor.add(Duration(minutes: structDur)),
        cluRemaining: dailyCluBudget * 0.20,
      ));
      cursor = cursor.add(Duration(minutes: structDur + 20)); // break

      // Afternoon (20%)
      int aftDur = (totalMinutes * 0.20).round();
      slots.add(_TimeSlot(
        window: ScheduledWindow.afternoon,
        start: cursor, end: cursor.add(Duration(minutes: aftDur)),
        cluRemaining: dailyCluBudget * 0.20,
      ));
      cursor = cursor.add(Duration(minutes: aftDur + 30)); // break

      // Review Evening (15%)
      int revDur = (totalMinutes * 0.15).round();
      slots.add(_TimeSlot(
        window: ScheduledWindow.reviewEvening,
        start: cursor, end: cursor.add(Duration(minutes: revDur)),
        cluRemaining: dailyCluBudget * 0.15,
      ));
      cursor = cursor.add(Duration(minutes: revDur));

      // Light Evening (5%)
      int lightDur = (totalMinutes * 0.05).round();
      slots.add(_TimeSlot(
        window: ScheduledWindow.reviewEvening,
        start: cursor, end: cursor.add(Duration(minutes: lightDur)),
        cluRemaining: dailyCluBudget * 0.05,
      ));

      // ── Sort subjects by nearest exam priority ──
      List<String> prioritySubjects = List.from(subjects);
      try {
        final upcomingExams = ExamRepository.instance.getUpcomingExams();
        if (upcomingExams.isNotEmpty) {
          final subjectNextExamDays = <String, int>{};
          for (final exam in upcomingExams) {
            if (exam.daysRemaining > 0 &&
                (!subjectNextExamDays.containsKey(exam.subject) ||
                    exam.daysRemaining < subjectNextExamDays[exam.subject]!)) {
              subjectNextExamDays[exam.subject] = exam.daysRemaining;
            }
          }
          prioritySubjects.sort((a, b) {
            final aDays = subjectNextExamDays[a] ?? 999;
            final bDays = subjectNextExamDays[b] ?? 999;
            return aDays.compareTo(bDays);
          });
        }
      } catch (e) {
        debugPrint('DailyPlanService: Could not prioritize by exams - $e');
      }

      // Focus area boost
      if (focusAreas != null && focusAreas.isNotEmpty) {
        final focus = focusAreas.toLowerCase();
        prioritySubjects.sort((a, b) {
          final aMatch = focus.contains(a.toLowerCase());
          final bMatch = focus.contains(b.toLowerCase());
          if (aMatch && !bMatch) return -1;
          if (!aMatch && bMatch) return 1;
          return 0;
        });
      }

      // ── Topic generator per subject (CAIE-aligned) ──
      final topicsBySubject = <String, List<String>>{
        'Mathematics': ['Algebra & Functions', 'Calculus', 'Trigonometry', 'Probability & Statistics', 'Vectors & Mechanics'],
        'Physics': ['Mechanics', 'Waves & Optics', 'Electricity & Magnetism', 'Thermal Physics', 'Modern Physics'],
        'Chemistry': ['Organic Chemistry', 'Physical Chemistry', 'Inorganic Chemistry', 'Electrochemistry', 'Kinetics'],
        'Biology': ['Cell Biology', 'Genetics', 'Ecology', 'Human Physiology', 'Biochemistry'],
        'Economics': ['Microeconomics', 'Macroeconomics', 'International Trade', 'Market Failure'],
        'Computer Science': ['Algorithms & Data Structures', 'Databases', 'Networking', 'Programming Paradigms'],
        'English': ['Language Analysis', 'Creative Writing', 'Comprehension', 'Critical Thinking'],
        'Business Studies': ['Marketing', 'Finance', 'Operations', 'Strategy & Leadership'],
      };

      final prefixMap = <String, String>{
        'deep_work': 'Master', 'practice': 'Practice',
        'review': 'Review', 'past_paper': 'Past Paper -',
        'flashcards': 'Flashcards -', 'mock_exam': 'Mock Exam -',
        'command_word_drill': 'Command Drill -',
        'examiner_report': 'Examiner Notes -',
      };

      // Description builder helper
      String descFor(String tt, String subj) {
        switch (tt) {
          case 'deep_work': return 'Deep focus on core concepts for $subj. Take structured notes.';
          case 'practice': return 'Active recall and practice problems for $subj.';
          case 'review': return 'Review and consolidate understanding of $subj.';
          case 'flashcards': return 'Spaced repetition flashcards for $subj key terms.';
          case 'past_paper': return 'Timed past paper practice for $subj. Use mark-scheme after completing.';
          case 'mock_exam': return 'Strict timed mock exam for $subj. No mark-scheme until complete.';
          case 'command_word_drill': return 'Practice CAIE command words for $subj. Self-assess against mark-scheme language patterns.';
          case 'examiner_report': return 'Review examiner reports for common mistakes in $subj.';
          default: return 'Study session for $subj.';
        }
      }

      // ── Allocate tasks with cognitive load management ──
      final tasks = <DailyPlanTask>[];
      var slotIdx = 0;
      var cluUsed = 0.0;
      final subjectCounts = <String, int>{};
      String? lastIntensity;
      var taskNum = 0;

      // Round-robin through subjects (interleaving — same as backend)
      final subjectQueue = <String>[];
      for (var round = 0; round < 3; round++) {
        subjectQueue.addAll(prioritySubjects);
      }

      for (final subj in subjectQueue) {
        if (slotIdx >= slots.length || cluUsed >= dailyCluBudget) break;

        // Pick task type from phase mix
        final ttName = phaseMix[taskNum % phaseMix.length].key;
        final clu = cluMap[ttName] ?? 5.0;

        // Cognitive load guard
        if (cluUsed + clu > dailyCluBudget) continue;

        // Max tasks per subject
        subjectCounts[subj] = (subjectCounts[subj] ?? 0) + 1;
        if (subjectCounts[subj]! > maxTasksPerSubject) continue;

        // Intensity based on days to exam (same as backend ObjectiveScoringEngine)
        final intensityLabel = daysToExam <= 3
            ? IntensityLevel.red
            : (daysToExam <= 30 ? IntensityLevel.orange : IntensityLevel.blue);
        final baseScore = switch (intensityLabel) {
          IntensityLevel.red => 0.75 + (_random.nextDouble() * 0.25),
          IntensityLevel.orange => 0.45 + (_random.nextDouble() * 0.30),
          IntensityLevel.blue => 0.2 + (_random.nextDouble() * 0.25),
        };
        final intensityScore = baseScore.clamp(0.0, 1.0);

        // No back-to-back RED (same as backend SlotManager)
        final effectiveIntensity = (lastIntensity == 'red' && intensityLabel == IntensityLevel.red)
            ? IntensityLevel.orange : intensityLabel;

        // Find available slot
        _TimeSlot? slot;
        for (var j = slotIdx; j < slots.length; j++) {
          if (slots[j].cluRemaining >= clu && slots[j].start.isBefore(slots[j].end)) {
            slot = slots[j];
            slotIdx = j + 1;
            break;
          }
        }
        if (slot == null) break;

        // Consume CLU
        slot.cluRemaining -= clu;
        cluUsed += clu;
        lastIntensity = effectiveIntensity.value;

        // Build task
        final taskType = TaskType.fromString(ttName);
        final duration = durMap[ttName] ?? 40;
        final taskStart = slot.start;
        final taskEnd = taskStart.add(Duration(minutes: duration));

        final topics = topicsBySubject[subj] ?? ['Core Concepts', 'Advanced Topics', 'Problem Solving'];
        final topic = topics[taskNum % topics.length];
        final prefix = prefixMap[ttName] ?? 'Study';
        final description = descFor(ttName, subj);

        final stableId = _hashId('$today|$subj|$ttName|${taskStart.millisecondsSinceEpoch}');

        final priority = switch (effectiveIntensity) {
          IntensityLevel.red => Priority.high,
          IntensityLevel.orange => Priority.medium,
          IntensityLevel.blue => Priority.low,
        };

        tasks.add(DailyPlanTask(
          id: 'plan_$stableId',
          title: '$prefix $topic',
          subject: subj,
          description: description,
          paper: ttName.contains('paper') || ttName.contains('mock') ? 'Paper 1' : '',
          date: today,
          startTime: taskStart,
          endTime: taskEnd,
          status: TaskStatus.pending,
          reason: '${(intensityScore * 100).toStringAsFixed(0)}% urgency | $phaseLabel | ${daysToExam.clamp(0, 999)}d to exam | target A*',
          isSyncToGoogle: false,
          isCompleted: false,
          intensityScore: intensityScore,
          intensityLabel: effectiveIntensity,
          phase: phase,
          anchorDate: today,
          taskType: taskType,
          scheduledWindow: slot.window,
          priority: priority,
        ));

        // Advance slot cursor (Pomodoro gap after deep-focus tasks)
        final gap = (ttName == 'deep_work' || ttName == 'past_paper' || ttName == 'mock_exam') ? 10 : 5;
        slot.start = taskEnd.add(Duration(minutes: gap));
        taskNum++;
      }

      if (tasks.isEmpty) {
        debugPrint('DailyPlanService: No tasks generated (no time slots available?)');
        return;
      }

      // ── Save to offline cache first (always works) ──
      if (uid.isNotEmpty) {
        await _saveTasksToOffline(uid, today, tasks);
      }

      // ── Try Firestore (may fail on device due to DNS block) ──
      try {
        final batch = FirebaseFirestore.instance.batch();
        for (final task in tasks) {
          final docRef = _dailyPlanCollection(uid).doc(task.id);
          batch.set(docRef, task.toJson(), SetOptions(merge: true));
        }
        await batch.commit();
        debugPrint('DailyPlanService: saved ${tasks.length} tasks to Firestore');
      } catch (fe) {
        debugPrint('DailyPlanService: Firestore save failed (using offline only) — $fe');
        // Offline cache already saved — this is fine
      }

      debugPrint(
          'DailyPlanService: generated ${tasks.length} smart local tasks '
          '(phase: $phaseLabel, CLU used: ${cluUsed.toStringAsFixed(1)}/$dailyCluBudget)');
    } catch (e) {
      debugPrint('DailyPlanService: smart local plan generation failed — $e');
    }
  }

  Future<int> _getDaysToNearestExam(List<String> subjects) async {
    try {
      final upcomingExams = ExamRepository.instance.getUpcomingExams();
      if (upcomingExams.isNotEmpty) {
        int minDays = 999;
        for (final exam in upcomingExams) {
          if (exam.daysRemaining > 0 && subjects.any((s) =>
              s.toLowerCase().contains(exam.subject.toLowerCase()) ||
              exam.subject.toLowerCase().contains(s.toLowerCase()))) {
            if (exam.daysRemaining < minDays) minDays = exam.daysRemaining;
          }
        }
        if (minDays < 999) return minDays;
      }
    } catch (_) {}
    return 999;
  }

  static String _hashId(String input) {
    final bytes = utf8.encode(input);
    final digest = md5.convert(bytes);
    return digest.toString().substring(0, 24);
  }

  Future<StudyPhase> _calculatePhase(List<String> subjects) async {
    try {
      final upcomingExams = ExamRepository.instance.getUpcomingExams();
      if (upcomingExams.isNotEmpty) {
        int minDays = 999;
        for (final exam in upcomingExams) {
          if (exam.daysRemaining > 0 && subjects.any((s) =>
              s.toLowerCase().contains(exam.subject.toLowerCase()) ||
              exam.subject.toLowerCase().contains(s.toLowerCase()))) {
            if (exam.daysRemaining < minDays) {
              minDays = exam.daysRemaining;
            }
          }
        }
        if (minDays < 999) {
          return StudyPhase.fromDays(minDays);
        }
      }
    } catch (_) {}
    return StudyPhase.foundation;
  }

  Future<List<String>> _getUserSubjects() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final subjects = prefs.getStringList('userSubjects');
      if (subjects != null && subjects.isNotEmpty) {
        return subjects;
      }
      final selectedSubjects = prefs.getStringList('selected_subjects');
      if (selectedSubjects != null && selectedSubjects.isNotEmpty) {
        return selectedSubjects;
      }
      for (final key in const ['userSubjects', 'selected_subjects']) {
        final raw = prefs.getString(key);
        if (raw == null || raw.trim().isEmpty) continue;
        final decoded = jsonDecode(raw);
        if (decoded is List) {
          final parsed = decoded
              .map((item) => item.toString().trim())
              .where((item) => item.isNotEmpty)
              .toList();
          if (parsed.isNotEmpty) return parsed;
        }
      }
      return [
        'Mathematics',
        'Physics',
        'Chemistry',
        'Biology',
        'Computer Science',
        'Economics',
      ];
    } catch (_) {
      return ['Mathematics', 'Physics'];
    }
  }

  Future<void> runDailyBuild() async {
    await _post('/planner/daily-build', retries: 1);
  }

  Future<void> rescheduleMissedBlock(String taskId) async {
    await _post('/planner/reschedule-missed-block', body: {'task_id': taskId}, retries: 1);
  }

  Future<List<DailyPlanTask>> ensureTodayPlan(String uid,
      {String? focusAreas}) async {
    final today = _todayString();
    final key = '$uid:$today:${focusAreas ?? ''}';
    final existing = await getTasksForDate(uid, today);
    if (existing.isNotEmpty) {
      return existing;
    }

    final current = _inFlightGenerations[key];
    if (current != null) return current;

    final future = _ensureTodayPlanInternal(uid, today, focusAreas: focusAreas);
    _inFlightGenerations[key] = future;
    future.whenComplete(() => _inFlightGenerations.remove(key));
    return future;
  }

  Future<List<DailyPlanTask>> _ensureTodayPlanInternal(
    String uid,
    String today, {
    String? focusAreas,
  }) async {
    // Try v2 endpoint first (no Firestore dependency - responds in <5s)
    try {
      final tasks = await _generateV2Plan(focusAreas: focusAreas);
      if (tasks.isNotEmpty) {
        debugPrint('DailyPlanService: v2 generated ${tasks.length} tasks');
        return tasks;
      }
      debugPrint(
          'DailyPlanService: v2 returned no tasks, using smart local fallback');
    } catch (e) {
      debugPrint('DailyPlanService: v2 generation failed - $e, using smart local');
    }

    // Fall back to smart local (offline, no network needed)
    await _generateSmartLocalPlan(focusAreas: focusAreas, uidOverride: uid);
    final tasks = await _loadTasksFromOffline(uid, today);
    if (tasks.isNotEmpty) {
      debugPrint('DailyPlanService: smart local generated ${tasks.length} tasks');
      return tasks;
    }

    // Last resort: check if subjects exist (don't require targetHours)
    final subjects = await _getUserSubjects();
    if (subjects.isEmpty) {
      throw const DailyPlanException(
        'Add subjects to generate a daily plan.',
      );
    }

    throw const DailyPlanException(
      'Daily planner could not create a plan. Add subjects or exam dates, then try again.',
    );
  }

  Future<http.Response> _post(
    String path, {
    Map<String, dynamic> body = const {},
    int retries = 2,
  }) async {
    final String? token;
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw const DailyPlanException('No authenticated user.');
      }
      token = await user.getIdToken();
    } catch (e) {
      debugPrint('DailyPlanService: failed to get ID token for $path — $e');
      throw DailyPlanException(
          'Could not authenticate daily planner request: $e');
    }

    if (token == null || token.isEmpty) {
      debugPrint('DailyPlanService: empty token, skipping $path');
      throw const DailyPlanException(
          'Could not authenticate daily planner request.');
    }

    final uid = FirebaseAuth.instance.currentUser?.uid ?? '';
    final payload = {'user_id': uid, ...body};

    for (int attempt = 0; attempt <= retries; attempt++) {
      try {
        final response = await _client
            .post(
              Uri.parse('$_backendUrl$path'),
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer $token',
              },
              body: jsonEncode(payload),
            )
            .timeout(_requestTimeout);

        if (response.statusCode < 200 || response.statusCode >= 300) {
          debugPrint(
            'DailyPlanService: $path returned ${response.statusCode}: ${response.body}',
          );
          if (attempt < retries) {
            final delay = Duration(seconds: attempt == 0 ? 3 : 8);
            debugPrint(
                'DailyPlanService: retrying $path (${attempt + 1}/$retries) in ${delay.inSeconds}s');
            await Future.delayed(delay);
            continue;
          }
          throw DailyPlanException(
            'Daily planner failed (${response.statusCode}).',
          );
        }
        return response;
      } on Exception catch (e) {
        debugPrint('DailyPlanService: network error on $path — $e');
        if (attempt < retries) {
          final delay = Duration(seconds: attempt == 0 ? 3 : 8);
          debugPrint(
              'DailyPlanService: retrying $path (${attempt + 1}/$retries) in ${delay.inSeconds}s');
          await Future.delayed(delay);
          continue;
        }
        if (e is DailyPlanException) rethrow;
        throw DailyPlanException('Daily planner network error: $e');
      }
    }
    throw const DailyPlanException('Daily planner request failed.');
  }

  Future<List<DailyPlanTask>> _waitForPlan(String uid, String date) async {
    const delays = [
      Duration(seconds: 1),
      Duration(seconds: 2),
      Duration(seconds: 3),
      Duration(seconds: 5),
      Duration(seconds: 8),
    ];

    for (final delay in delays) {
      await Future<void>.delayed(delay);
      final tasks = await getTasksForDate(uid, date);
      if (tasks.isNotEmpty) return tasks;
    }

    return getTasksForDate(uid, date);
  }

  String _todayString([DateTime? now]) =>
      (now ?? DateTime.now()).toIso8601String().split('T').first;

  // ========== OFFLINE STORAGE ==========
  static const String _offlinePlansKey = 'offline_daily_plans';

  Future<Map<String, List<Map<String, dynamic>>>> _loadOfflinePlans() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final data = prefs.getString(_offlinePlansKey);
      if (data == null || data.isEmpty) return {};
      final decoded = jsonDecode(data) as Map<String, dynamic>;
      return decoded.map((k, v) => MapEntry(
        k,
        (v as List).map((e) => Map<String, dynamic>.from(e as Map)).toList(),
      ));
    } catch (e) {
      debugPrint('DailyPlanService: _loadOfflinePlans failed - $e');
      return {};
    }
  }

  Future<void> _saveOfflinePlans(Map<String, List<Map<String, dynamic>>> plans) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final data = jsonEncode(plans);
      await prefs.setString(_offlinePlansKey, data);
    } catch (e) {
      debugPrint('DailyPlanService: _saveOfflinePlans failed - $e');
    }
  }

  Future<void> _saveTasksToOffline(
    String uid,
    String date,
    List<DailyPlanTask> tasks,
  ) async {
    final all = await _loadOfflinePlans();
    final key = '${uid}_$date';
    all[key] = tasks.map((t) => t.toJson()).toList();
    await _saveOfflinePlans(all);
    debugPrint('DailyPlanService: saved ${tasks.length} tasks for $key to offline');
  }

  Future<List<DailyPlanTask>> _loadTasksFromOffline(String uid, String date) async {
    final all = await _loadOfflinePlans();
    final key = '${uid}_$date';
    final taskMaps = all[key];
    if (taskMaps == null) return [];
    return taskMaps.map((t) => DailyPlanTask.fromJson(t['id'] as String, t)).toList();
  }

  // ========== USER DATA CHECK ==========
  Future<bool> checkUserDataSufficient() async {
    final subjects = await _getUserSubjects();
    if (subjects.isEmpty) return false;
    
    try {
      final prefs = await SharedPreferences.getInstance();
      final targetHours = prefs.getDouble('userTargetHours') ?? 0;
      return targetHours >= 1;
    } catch (_) {
      return subjects.isNotEmpty;
    }
  }

  // ========== MONTHLY PLAN ==========
  /// Auto-generate monthly plan (30 days). Returns empty for new users without enough data.
  /// Starts offline and online generation simultaneously.
  Future<List<List<DailyPlanTask>>> ensureMonthPlan({
    required String uid,
    String? focusAreas,
  }) async {
    // Check user data first
    final hasData = await checkUserDataSufficient();
    if (!hasData) {
      debugPrint('DailyPlanService: insufficient user data for monthly plan');
      return [];
    }
    
    final results = <List<DailyPlanTask>>[];
    final today = DateTime.now();
    
    for (int i = 0; i < 30; i++) {
      final date = today.add(Duration(days: i));
      final dateStr = date.toIso8601String().split('T').first;
      
      // Check if we already have this date's plan (Firestore)
      final existing = await getTasksForDate(uid, dateStr);
      if (existing.isNotEmpty) {
        results.add(existing);
        continue;
      }
      
      // Also check offline cache
      final offlineTasks = await _loadTasksFromOffline(uid, dateStr);
      if (offlineTasks.isNotEmpty) {
        results.add(offlineTasks);
        // Sync to Firestore in background
        _syncOfflineToFirestore(uid, dateStr, offlineTasks);
        continue;
      }
      
      // Start offline AND online simultaneously
      final onlineTask = _generateOnlineForDate(uid, dateStr, focusAreas);
      
      // Use offline plan or wait for online
      List<DailyPlanTask> plan = offlineTasks.isNotEmpty 
          ? offlineTasks 
          : await onlineTask;
      
      if (plan.isNotEmpty) {
        results.add(plan);
        // Save to offline storage
        await _saveTasksToOffline(uid, dateStr, plan);
      }
    }
    
    return results;
  }

  Future<void> _syncOfflineToFirestore(String uid, String date, List<DailyPlanTask> tasks) async {
    try {
      final batch = FirebaseFirestore.instance.batch();
      for (final task in tasks) {
        final ref = _dailyPlanCollection(uid).doc(task.id);
        batch.set(ref, task.toJson());
      }
      await batch.commit();
    } catch (e) {
      debugPrint('DailyPlanService: _syncOfflineToFirestore failed - $e');
    }
  }

  Future<List<DailyPlanTask>> _generateOnlineForDate(
    String uid,
    String date,
    String? focusAreas,
  ) async {
    try {
      final response = await _post(
        '/generate-daily-plan',
        body: {
          'user_id': uid,
          'client_date': date,
          if (focusAreas != null) 'focus_areas': focusAreas,
          'force': true,
        },
        retries: 1,
      );
      return _tasksFromResponse(response);
    } catch (e) {
      debugPrint('DailyPlanService: _generateOnlineForDate failed - $e');
      return [];
    }
  }

  List<DailyPlanTask> _tasksFromResponse(http.Response response) {
    if (response.statusCode != 200) return [];
    try {
      final data = jsonDecode(response.body);
      final taskList = data['tasks'] as List?;
      if (taskList == null) return [];
      return taskList
          .map((t) => DailyPlanTask.fromJson(t['id'] as String? ?? '', t as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  List<DailyPlanTask> _sortTasks(List<DailyPlanTask> tasks) {
    tasks.sort((a, b) {
      final dateCompare = a.date.compareTo(b.date);
      if (dateCompare != 0) return dateCompare;
      return a.startTime.compareTo(b.startTime);
    });
    return tasks;
  }

  int? _taskCountFromResponse(http.Response response) {
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic>) {
        final count = decoded['task_count'];
        if (count is num) return count.toInt();
      }
    } catch (_) {}
    return null;
  }

  Future<void> updateTask(DailyPlanTask task) async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) return;

      final data = <String, dynamic>{
        'is_completed': task.isCompleted,
        'status': task.status.value,
      };
      if (task.isCompleted) {
        data['completed_at'] = DateTime.now().toIso8601String();
      }
      await _dailyPlanCollection(user.uid).doc(task.id).update(data);
    } catch (e) {
      debugPrint('DailyPlanService: failed to update task — $e');
    }
  }

  Future<void> updateTaskTimes(String uid, String taskId, DateTime newStartTime, DateTime newEndTime) async {
    try {
      final docRef = _dailyPlanCollection(uid).doc(taskId);
      await docRef.update({
        'start_time': newStartTime.toIso8601String(),
        'end_time': newEndTime.toIso8601String(),
      });
      debugPrint('DailyPlanService: updated task times for $taskId');
    } catch (e) {
      debugPrint('DailyPlanService: failed to update task times — $e');
    }
  }
}

/// Internal time slot for cognitive-load-aware scheduling
class _TimeSlot {
  ScheduledWindow window;
  DateTime start;
  DateTime end;
  double cluRemaining;

  _TimeSlot({
    required this.window,
    required this.start,
    required this.end,
    required this.cluRemaining,
  });
}
