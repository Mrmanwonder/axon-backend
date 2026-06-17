// lib/providers/plan_provider.dart
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/daily_plan_task.dart';
import '../services/daily_plan_service.dart';
import 'auth_provider.dart';

final dailyPlanServiceProvider = Provider((ref) => DailyPlanService());

final todayPlanProvider = StreamProvider<List<DailyPlanTask>>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  final auth = ref.watch(authStateProvider);
  if (!auth.isAuthenticated) return const Stream.empty();
  return service.watchTodayPlan(auth.user!.uid);
});

final generatePlanProvider =
    Provider<Future<void> Function(String? focusAreas)>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  return (String? focusAreas) =>
      service.generateTodayPlan(focusAreas: focusAreas);
});

final ensureTodayPlanProvider =
    Provider<Future<List<DailyPlanTask>> Function(String? focusAreas)>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  final auth = ref.watch(authStateProvider);
  if (!auth.isAuthenticated) return (_) async => [];
  return (String? focusAreas) =>
      service.ensureTodayPlan(auth.user!.uid, focusAreas: focusAreas);
});

final studyStreakProvider = FutureProvider<int>((ref) async {
  final prefs = await SharedPreferences.getInstance();
  final raw = prefs.getString('timer_history');
  if (raw == null || raw.isEmpty) return 0;

  try {
    final decoded = jsonDecode(raw);
    if (decoded is! List) return 0;

    final days = decoded
        .whereType<Map>()
        .map((entry) => DateTime.tryParse((entry['date'] ?? '').toString()))
        .whereType<DateTime>()
        .map((date) => DateTime(date.year, date.month, date.day))
        .toSet()
        .toList()
      ..sort((a, b) => b.compareTo(a));

    if (days.isEmpty) return 0;

    var streak = 0;
    var cursor = DateTime.now();
    cursor = DateTime(cursor.year, cursor.month, cursor.day);

    for (final day in days) {
      final diff = cursor.difference(day).inDays;
      if (diff > 1) break;
      streak++;
      cursor = day.subtract(const Duration(days: 1));
    }
    return streak;
  } catch (_) {
    return 0;
  }
});
