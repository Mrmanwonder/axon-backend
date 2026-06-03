import 'dart:convert';

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
      final response = await _post(
        '/generate-daily-plan',
        body: {
          if (focusAreas != null && focusAreas.isNotEmpty)
            'focus_areas': focusAreas,
          'client_date': _todayString(),
          'force': true,
        },
        retries: 2,
      );
      if (_taskCountFromResponse(response) == 0) {
        debugPrint(
            'DailyPlanService: remote force generation returned no tasks, using local fallback');
        await _generateLocalPlan(focusAreas: focusAreas);
      }
    } catch (e) {
      debugPrint(
          'DailyPlanService: remote generation failed, using local fallback — $e');
      await _generateLocalPlan(focusAreas: focusAreas);
    } finally {
      _isGenerating = false;
    }
  }

  Future<void> _generateLocalPlan({
    String? focusAreas,
    String? uidOverride,
  }) async {
    debugPrint('DailyPlanService: _generateLocalPlan() called');
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

      final now = DateTime.now();
      final tasks = <DailyPlanTask>[];

      final totalBlocks = targetHours.ceil();

      // Calculate phase from nearest exam
      final phase = await _calculatePhase(subjects);

      // Slot scheduling: 60% morning/peak, 40% afternoon/evening
      int startHour = now.hour;
      if (now.minute > 30) startHour++;
      if (startHour < 8) startHour = 8;

      final morningSlots = <Map<String, String>>[];
      final afternoonSlots = <Map<String, String>>[];

      for (var h = startHour; h < 23; h++) {
        final sh = h.toString().padLeft(2, '0');
        final eh = (h + 1).toString().padLeft(2, '0');
        final slot = {'start': '$sh:00', 'end': '$eh:00'};
        if (h < 14) {
          morningSlots.add(slot);
        } else {
          afternoonSlots.add(slot);
        }
      }

      // Allocate 60% to morning, 40% to afternoon
      final morningCount = (totalBlocks * 0.6).ceil();
      final afternoonCount = totalBlocks - morningCount;

      final slots = <Map<String, String>>[];
      for (var i = 0; i < morningCount && i < morningSlots.length; i++) {
        slots.add(morningSlots[i]);
      }
      for (var i = 0; i < afternoonCount && i < afternoonSlots.length; i++) {
        slots.add(afternoonSlots[i]);
      }

      if (slots.isEmpty) {
        debugPrint('DailyPlanService: No time left today to schedule tasks.');
        return;
      }

      // Sort subjects by nearest exam
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

      // Cycle of task types for diversity
      const taskTypeCycle = [
        TaskType.deepWork,
        TaskType.practice,
        TaskType.deepWork,
        TaskType.flashcards,
        TaskType.review,
        TaskType.practice,
        TaskType.pastPaper,
        TaskType.review,
      ];

      for (var i = 0; i < slots.length; i++) {
        final subject = prioritySubjects[i % prioritySubjects.length];
        final slot = slots[i];
        final hour = int.parse(slot['start']!.split(':')[0]);

        // Morning slots = high intensity, afternoon = medium, late = low
        final isPeak = hour < 12;
        final isAfternoon = hour >= 12 && hour < 17;
        final double intensityScore;
        final IntensityLevel intensityLabel;
        if (isPeak) {
          intensityScore = 0.9;
          intensityLabel = IntensityLevel.red;
        } else if (isAfternoon) {
          intensityScore = 0.6;
          intensityLabel = IntensityLevel.orange;
        } else {
          intensityScore = 0.3;
          intensityLabel = IntensityLevel.blue;
        }

        final taskType = taskTypeCycle[i % taskTypeCycle.length];

        final ScheduledWindow window;
        if (hour < 12) {
          window = hour < 10 ? ScheduledWindow.peakFocusMorning : ScheduledWindow.morning;
        } else if (hour < 17) {
          window = ScheduledWindow.afternoon;
        } else {
          window = ScheduledWindow.reviewEvening;
        }

        final descriptions = {
          TaskType.deepWork: 'Deep focus session on new concepts for $subject',
          TaskType.practice: 'Active recall and practice questions for $subject',
          TaskType.review: 'Review notes and consolidate understanding of $subject',
          TaskType.flashcards: 'Spaced repetition flashcards for $subject',
          TaskType.pastPaper: 'Timed past paper practice for $subject',
          TaskType.mockExam: 'Mock exam simulation for $subject',
          TaskType.commandWordDrill: 'Command word drill for $subject',
        };

        final titleByType = {
          TaskType.deepWork: '$subject — Deep Work',
          TaskType.practice: '$subject — Practice',
          TaskType.review: '$subject — Review',
          TaskType.flashcards: '$subject — Flashcards',
          TaskType.pastPaper: '$subject — Past Paper',
          TaskType.mockExam: '$subject — Mock Exam',
          TaskType.commandWordDrill: '$subject — Command Drill',
        };

        final reasons = [
          'Scheduled based on your daily study target of ${targetHours.toStringAsFixed(1)} hours',
          'Distributed practice helps improve long-term retention',
          'Aligned with your upcoming exam schedule',
          'Morning peak focus window for high-intensity work',
          'Afternoon consolidation of morning concepts',
        ];

        final subjectSlug = subject
            .toLowerCase()
            .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
            .replaceAll(RegExp(r'^_|_$'), '');
        final taskId =
            'axon_plan_${today.replaceAll('-', '')}_${i}_$subjectSlug';

        final todayDate = DateTime.parse(today);
        final startParts = slot['start']!.split(':');

        final startTime = DateTime(todayDate.year, todayDate.month,
            todayDate.day, int.parse(startParts[0]), int.parse(startParts[1]));
        final endTime = startTime.add(const Duration(minutes: 50));

        tasks.add(DailyPlanTask(
          id: taskId,
          title: titleByType[taskType] ?? '$subject Session',
          subject: subject,
          description: descriptions[taskType] ?? 'Study session for $subject',
          date: today,
          startTime: startTime,
          endTime: endTime,
          status: TaskStatus.pending,
          reason: reasons[i % reasons.length],
          isSyncToGoogle: false,
          isCompleted: false,
          intensityScore: intensityScore,
          intensityLabel: intensityLabel,
          phase: phase,
          anchorDate: today,
          taskType: taskType,
          scheduledWindow: window,
          priority: i == 0
              ? Priority.high
              : (i <= 2 ? Priority.medium : Priority.low),
        ));
      }

      final batch = FirebaseFirestore.instance.batch();
      for (final task in tasks) {
        final docRef = _dailyPlanCollection(uid).doc(task.id);
        batch.set(docRef, task.toJson(), SetOptions(merge: true));
      }
      await batch.commit();
      debugPrint(
          'DailyPlanService: generated ${tasks.length} intelligent local tasks (phase: ${phase.label})');
    } catch (e) {
      debugPrint('DailyPlanService: local plan generation failed — $e');
    }
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
    try {
      final response = await _post(
        '/generate-daily-plan',
        body: {
          if (focusAreas != null && focusAreas.isNotEmpty)
            'focus_areas': focusAreas,
          'client_date': today,
          'force': false,
        },
        retries: 2,
      );
      if (_taskCountFromResponse(response) == 0) {
        debugPrint(
            'DailyPlanService: planner returned no tasks, using local fallback');
        await _generateLocalPlan(focusAreas: focusAreas, uidOverride: uid);
      }
    } catch (e) {
      debugPrint('DailyPlanService: ensureTodayPlan backend failed - $e');
      await _generateLocalPlan(focusAreas: focusAreas, uidOverride: uid);
    }

    final tasks = await _waitForPlan(uid, today);
    if (tasks.isNotEmpty) return tasks;

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
}
