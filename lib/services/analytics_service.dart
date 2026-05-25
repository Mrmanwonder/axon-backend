// lib/services/analytics_service.dart
// ─────────────────────────────────────────────────────────────────
// Advanced Analytics Service
// Subject performance, correlations, heatmaps, insights
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'dart:math';
import 'package:shared_preferences/shared_preferences.dart';

class AnalyticsService {
  static final AnalyticsService _instance = AnalyticsService._internal();
  factory AnalyticsService() => _instance;
  static AnalyticsService get instance => _instance;
  AnalyticsService._internal();

  static const String _sessionHistoryKey = 'timer_history';
  static const String _mockScoresKey = 'mock_scores_history';

  DateTime _safeParseDate(String? dateStr, [DateTime? fallback]) {
    try {
      if (dateStr == null || dateStr.isEmpty) return fallback ?? DateTime.now();
      return DateTime.parse(dateStr);
    } catch (_) {
      return fallback ?? DateTime.now();
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // SUBJECT PERFORMANCE
  // ─────────────────────────────────────────────────────────────────

  Future<List<SubjectPerformance>> getSubjectPerformance() async {
    final sessions = await _getSessionHistory();
    final mockScores = await _getMockScores();

    final subjectData = <String, SubjectPerformanceData>{};

    for (final session in sessions) {
      final subject = session['subject'] ?? 'Unknown';
      final duration = (session['duration'] ?? 0) as int;
      final intensity = (session['intensity'] ?? 0.0).toDouble();
      final focusQuality = (session['focusQuality'] ?? 0.0).toDouble();

      if (!subjectData.containsKey(subject)) {
        subjectData[subject] = SubjectPerformanceData(
          subject: subject,
          totalMinutes: 0,
          sessionCount: 0,
          totalIntensity: 0,
          totalFocusQuality: 0,
          chaptersStudied: [],
        );
      }

      final current = subjectData[subject]!;
      subjectData[subject] = SubjectPerformanceData(
        subject: subject,
        totalMinutes: current.totalMinutes + (duration ~/ 60),
        sessionCount: current.sessionCount + 1,
        totalIntensity: current.totalIntensity + intensity,
        totalFocusQuality: current.totalFocusQuality + focusQuality,
        chaptersStudied: _mergeChapters(
            current.chaptersStudied, session['chapter']?.toString() ?? ''),
      );
    }

    for (final entry in subjectData.entries) {
      final current = entry.value;
      final mockData = mockScores
          .where((s) => s['subject'] == entry.key)
          .toList()
        ..sort((a, b) => (a['date'] as String).compareTo(b['date'] as String));

      double trend = 0;
      double averageMockScore = 0;
      if (mockData.length >= 2) {
        final first = (mockData.first['score'] as num).toDouble();
        final last = (mockData.last['score'] as num).toDouble();
        trend = (last - first) / 100;
        averageMockScore = mockData
                .map((s) => (s['score'] as num).toDouble())
                .reduce((a, b) => a + b) /
            mockData.length;
      } else if (mockData.isNotEmpty) {
        averageMockScore = (mockData.first['score'] as num).toDouble();
      }

      entry.value.averageIntensity =
          current.totalIntensity / current.sessionCount;
      entry.value.averageFocusQuality =
          current.totalFocusQuality / current.sessionCount;
      entry.value.trend = trend;
      entry.value.averageMockScore = averageMockScore;
    }

    return subjectData.values
        .map((d) => SubjectPerformance(
              subject: d.subject,
              totalSessions: d.sessionCount,
              totalMinutes: d.totalMinutes,
              averageScore: d.averageMockScore,
              dailyPerformances: [],
              strongDays: d.strongChapters.length,
              weakDays: d.weakChapters.length,
              averageIntensity: d.averageIntensity,
              averageFocusQuality: d.averageFocusQuality,
              chaptersStudied: d.chaptersStudied,
              weakChapters: d.weakChapters,
              strongChapters: d.strongChapters,
              trend: d.trend,
            ))
        .toList()
      ..sort((a, b) => b.totalMinutes.compareTo(a.totalMinutes));
  }

  List<String> _mergeChapters(List<String> chapters, String newChapter) {
    if (newChapter.isEmpty) return chapters;
    if (chapters.contains(newChapter)) return chapters;
    return [...chapters, newChapter];
  }

  // ─────────────────────────────────────────────────────────────────
  // TIME VS SCORE CORRELATION
  // ─────────────────────────────────────────────────────────────────

  Future<CorrelationAnalysis> getTimeScoreCorrelation() async {
    final sessions = await _getSessionHistory();
    final mockScores = await _getMockScores();

    final weeklyStudy = <String, double>{};
    final weeklyScores = <String, double>{};

    for (final session in sessions) {
      final date = _safeParseDate(session['date']);
      final week = _getWeekKey(date);
      final minutes = (session['duration'] ?? 0) / 60.0;
      weeklyStudy[week] = (weeklyStudy[week] ?? 0) + minutes;
    }

    for (final score in mockScores) {
      final date = _safeParseDate(score['date']);
      final week = _getWeekKey(date);
      final scoreVal = (score['score'] ?? 0).toDouble();
      weeklyScores[week] = scoreVal;
    }

    final weeks = weeklyStudy.keys.toList()..sort();
    if (weeks.length < 3) {
      return CorrelationAnalysis(
        correlation: 0,
        slope: 0,
        intercept: 0,
        weeklyData: [],
        insight: 'Not enough data for correlation analysis',
        isSignificant: false,
      );
    }

    final studyValues = weeks.map((w) => weeklyStudy[w] ?? 0).toList();
    final scoreValues = weeks.map((w) => weeklyScores[w] ?? 0).toList();

    final validIndices = <int>[];
    for (int i = 0; i < weeks.length; i++) {
      if (scoreValues[i] > 0) validIndices.add(i);
    }

    if (validIndices.length < 3) {
      return CorrelationAnalysis(
        correlation: 0,
        slope: 0,
        intercept: 0,
        weeklyData: [],
        insight: 'Not enough mock scores for correlation',
        isSignificant: false,
      );
    }

    final validStudy = validIndices.map((i) => studyValues[i]).toList();
    final validScore = validIndices.map((i) => scoreValues[i]).toList();

    final correlation = _pearsonCorrelation(validStudy, validScore);
    final regression = _linearRegression(validStudy, validScore);

    final weeklyData = weeks.asMap().entries.map((e) {
      return WeeklyDataPoint(
        week: e.value,
        studyHours: studyValues[e.key],
        mockScore: scoreValues[e.key],
      );
    }).toList();

    String insight;
    if (correlation.abs() < 0.3) {
      insight =
          'Study time shows weak correlation with scores. Focus on study quality over quantity.';
    } else if (correlation > 0) {
      final improvement = regression.slope > 0
          ? 'For every extra hour of study, your score increases by ~${(regression.slope * 10).round() / 10} points'
          : 'You\'re getting good returns on your study time';
      insight =
          'Strong positive correlation (${(correlation * 100).round()}%). $improvement';
    } else {
      insight =
          'Unexpected negative correlation. Your study methods may need adjustment.';
    }

    return CorrelationAnalysis(
      correlation: correlation,
      slope: regression.slope,
      intercept: regression.intercept,
      weeklyData: weeklyData,
      insight: insight,
      isSignificant: correlation.abs() > 0.5,
    );
  }

  String _getWeekKey(DateTime date) {
    return '${date.year}-W${(date.day / 7).ceil().toString().padLeft(2, '0')}';
  }

  double _pearsonCorrelation(List<double> x, List<double> y) {
    final n = x.length;
    if (n == 0) return 0;

    final meanX = x.reduce((a, b) => a + b) / n;
    final meanY = y.reduce((a, b) => a + b) / n;

    var sumXY = 0.0;
    var sumX2 = 0.0;
    var sumY2 = 0.0;

    for (int i = 0; i < n; i++) {
      final dx = x[i] - meanX;
      final dy = y[i] - meanY;
      sumXY += dx * dy;
      sumX2 += dx * dx;
      sumY2 += dy * dy;
    }

    if (sumX2 == 0 || sumY2 == 0) return 0;
    return sumXY / sqrt(sumX2 * sumY2);
  }

  ({double slope, double intercept}) _linearRegression(
      List<double> x, List<double> y) {
    final n = x.length;
    if (n == 0) return (slope: 0, intercept: 0);

    final meanX = x.reduce((a, b) => a + b) / n;
    final meanY = y.reduce((a, b) => a + b) / n;

    var sumXY = 0.0;
    var sumX2 = 0.0;

    for (int i = 0; i < n; i++) {
      sumXY += (x[i] - meanX) * (y[i] - meanY);
      sumX2 += (x[i] - meanX) * (x[i] - meanX);
    }

    if (sumX2 == 0) return (slope: 0, intercept: meanY);

    final slope = sumXY / sumX2;
    final intercept = meanY - slope * meanX;

    return (slope: slope, intercept: intercept);
  }

  // ─────────────────────────────────────────────────────────────────
  // IGNORED / DELAYED CHAPTERS
  // ─────────────────────────────────────────────────────────────────

  Future<DelayedChaptersReport> getDelayedChapters() async {
    final sessions = await _getSessionHistory();
    if (sessions.isEmpty) {
      return const DelayedChaptersReport(
          delayed: [], ignored: [], totalDelayed: 0, totalIgnored: 0);
    }
    final catalog = await _loadStudyCatalog();

    final subjectChapters = <String, Set<String>>{};
    final studiedChapters = <String, Set<String>>{};
    final lastStudied = <String, DateTime>{};
    final studyCounts = <String, int>{};

    for (final session in sessions) {
      final subject = session['subject'] ?? 'Unknown';
      final chapter = session['chapter']?.toString() ?? '';
      final date = _safeParseDate(session['date']);

      studiedChapters[subject] ??= {};
      if (chapter.isNotEmpty) {
        studiedChapters[subject]!.add(chapter);
        studyCounts['${subject}_$chapter'] =
            (studyCounts['${subject}_$chapter'] ?? 0) + 1;
        final key = '${subject}_$chapter';
        if (!lastStudied.containsKey(key) || date.isAfter(lastStudied[key]!)) {
          lastStudied[key] = date;
        }
      }
    }

    for (final entry in catalog.entries) {
      subjectChapters[entry.key] = Set<String>.from(entry.value);
    }

    final delayed = <DelayedChapter>[];
    final ignored = <DelayedChapter>[];

    for (final subject in subjectChapters.keys) {
      final expected = subjectChapters[subject] ?? {};
      final studied = studiedChapters[subject] ?? {};
      final notStudied = expected.difference(studied);

      for (final chapter in notStudied) {
        ignored.add(DelayedChapter(
          subject: subject,
          chapter: chapter,
          daysSinceStudied: null,
          studyCount: 0,
          isIgnored: true,
          priority: _estimatePriority(chapter),
        ));
      }

      for (final chapter in studied) {
        final key = '${subject}_$chapter';
        final days = lastStudied[key] != null
            ? DateTime.now().difference(lastStudied[key]!).inDays
            : 0;
        final count = studyCounts[key] ?? 0;

        if (days > 14 || count < 2) {
          delayed.add(DelayedChapter(
            subject: subject,
            chapter: chapter,
            daysSinceStudied: days,
            studyCount: count,
            isIgnored: false,
            priority: _estimatePriority(chapter),
          ));
        }
      }
    }

    delayed.sort((a, b) => b.priority.compareTo(a.priority));
    ignored.sort((a, b) => b.priority.compareTo(a.priority));

    return DelayedChaptersReport(
      delayed: delayed.take(10).toList(),
      ignored: ignored.take(10).toList(),
      totalDelayed: delayed.length,
      totalIgnored: ignored.length,
    );
  }

  int _estimatePriority(String chapter) {
    final lower = chapter.toLowerCase();
    if (lower.contains('introduction') || lower.contains('basics')) return 1;
    if (lower.contains('advanced') || lower.contains('complex')) return 3;
    return 2;
  }

  Future<Map<String, List<String>>> _loadStudyCatalog() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('study_catalog');
    if (raw == null || raw.isEmpty) return {};

    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;
      return data.map((k, v) => MapEntry(k, List<String>.from(v)));
    } catch (_) {
      return {};
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // STUDY HEATMAP
  // ─────────────────────────────────────────────────────────────────

  Future<StudyHeatmap> getStudyHeatmap() async {
    final sessions = await _getSessionHistory();

    final byDayOfWeek = <int, double>{};
    final byHour = <int, double>{};
    final byDayAndHour = <int, Map<int, double>>{};
    final intensityByDayHour = <int, Map<int, double>>{};
    final hoursPerDay = <int, int>{};

    for (int i = 0; i < 7; i++) {
      byDayOfWeek[i] = 0;
      byDayAndHour[i] = {};
      intensityByDayHour[i] = {};
      hoursPerDay[i] = 0;
      for (int h = 0; h < 24; h++) {
        byDayAndHour[i]![h] = 0;
        intensityByDayHour[i]![h] = 0;
      }
    }

    for (int h = 0; h < 24; h++) {
      byHour[h] = 0;
    }

    for (final session in sessions) {
      final date = _safeParseDate(session['date']);
      final hours = (session['duration'] ?? 0) / 60.0;
      final intensity = (session['intensity'] ?? 0.0).toDouble();
      final hour = date.hour;
      final dayOfWeek = (date.weekday - 1) % 7;

      byDayOfWeek[dayOfWeek] = (byDayOfWeek[dayOfWeek] ?? 0) + hours;
      byHour[hour] = (byHour[hour] ?? 0) + hours;
      hoursPerDay[dayOfWeek] = (hoursPerDay[dayOfWeek] ?? 0) + 1;

      byDayAndHour[dayOfWeek]![hour] =
          (byDayAndHour[dayOfWeek]![hour] ?? 0) + hours;
      final currentInt = intensityByDayHour[dayOfWeek]![hour]!;
      final prevHours = byDayAndHour[dayOfWeek]![hour]! - hours;
      intensityByDayHour[dayOfWeek]![hour] = prevHours > 0
          ? (currentInt * (hoursPerDay[dayOfWeek]! - 1) + intensity) /
              hoursPerDay[dayOfWeek]!
          : intensity;
    }

    final maxHour = byHour.values.isEmpty
        ? 1.0
        : byHour.values.reduce((a, b) => a > b ? a : b);
    final normalizedByHour =
        byHour.map((k, v) => MapEntry(k, maxHour > 0 ? v / maxHour : 0.0));

    final maxDay = byDayOfWeek.values.isEmpty
        ? 1.0
        : byDayOfWeek.values.reduce((a, b) => a > b ? a : b);
    final normalizedByDay =
        byDayOfWeek.map((k, v) => MapEntry(k, maxDay > 0 ? v / maxDay : 0.0));

    int peakHour = 0;
    double maxHourVal = 0;
    byHour.forEach((h, v) {
      if (v > maxHourVal) {
        maxHourVal = v;
        peakHour = h;
      }
    });

    int peakDay = 0;
    double maxDayVal = 0;
    byDayOfWeek.forEach((d, v) {
      if (v > maxDayVal) {
        maxDayVal = v;
        peakDay = d;
      }
    });

    return StudyHeatmap(
      byDayOfWeek: byDayOfWeek,
      byHour: byHour,
      byDayAndHour: byDayAndHour,
      intensityByDayHour: intensityByDayHour,
      normalizedByDayOfWeek: normalizedByDay,
      normalizedByHour: normalizedByHour,
      peakHour: peakHour,
      peakDay: peakDay,
      totalStudyHours:
          sessions.fold(0.0, (sum, s) => sum + ((s['duration'] ?? 0) / 60)),
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // MOCK PAPER TRENDLINES
  // ─────────────────────────────────────────────────────────────────

  Future<MockTrendAnalysis> getMockTrendAnalysis() async {
    final mockScores = await _getMockScores();

    if (mockScores.length < 2) {
      return MockTrendAnalysis(
        trendLine: [],
        upperBound: [],
        lowerBound: [],
        dataPoints: [],
        averageScore: 0,
        improvement: 0,
        confidence: 0,
        predictions: [],
      );
    }

    mockScores
        .sort((a, b) => (a['date'] as String).compareTo(b['date'] as String));

    final scores = mockScores.map((s) => (s['score'] ?? 0).toDouble()).toList();
    final dates = mockScores.map((s) => _safeParseDate(s['date'])).toList();

    final n = scores.length;
    final xValues = List.generate(n, (i) => i.toDouble());
    final scoresDouble = scores.map((s) => s as double).toList();
    final regression = _linearRegression(xValues, scoresDouble);

    final trendLine = xValues
        .map((x) => regression.slope * x + regression.intercept)
        .toList();

    final residuals = <double>[];
    for (int i = 0; i < n; i++) {
      residuals.add(scores[i] - trendLine[i]);
    }
    final variance = residuals.map((r) => r * r).reduce((a, b) => a + b) / n;
    final stdDev = sqrt(variance);

    final upperBound = trendLine.map((y) => y + 1.96 * stdDev).toList();
    final lowerBound =
        trendLine.map((y) => (y - 1.96 * stdDev).clamp(0.0, 100.0)).toList();

    final dataPoints = mockScores
        .asMap()
        .entries
        .map((e) => MockDataPoint(
              date: dates[e.key],
              score: scores[e.key],
              subject: e.value['subject']?.toString() ?? 'General',
            ))
        .toList();

    final predictions = <PredictionPoint>[];
    for (int i = 1; i <= 3; i++) {
      final futureX = (n - 1 + i * 7).toDouble();
      final predicted = regression.slope * futureX + regression.intercept;
      predictions.add(PredictionPoint(
        date: dates.last.add(Duration(days: i * 7)),
        predictedScore: predicted.clamp(0, 100),
        confidence: (1 - (i * 0.15)).clamp(0.5, 0.95),
      ));
    }

    return MockTrendAnalysis(
      trendLine: trendLine,
      upperBound: upperBound,
      lowerBound: lowerBound,
      dataPoints: dataPoints,
      averageScore: scores.reduce((a, b) => a + b) / n,
      improvement: scores.last - scores.first,
      confidence: n >= 5 ? 0.85 : (n / 5 * 0.85),
      predictions: predictions,
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // WHAT CHANGED INSIGHTS
  // ─────────────────────────────────────────────────────────────────

  Future<WeeklyInsight> getWeeklyInsight() async {
    final sessions = await _getSessionHistory();

    final now = DateTime.now();
    final thisWeek =
        _getWeekSessions(sessions, now.subtract(const Duration(days: 7)), now);
    final lastWeek = _getWeekSessions(
        sessions,
        now.subtract(const Duration(days: 14)),
        now.subtract(const Duration(days: 7)));

    final thisWeekHours =
        thisWeek.fold(0.0, (sum, s) => sum + ((s['duration'] ?? 0) / 60));
    final lastWeekHours =
        lastWeek.fold(0.0, (sum, s) => sum + ((s['duration'] ?? 0) / 60));

    final hoursDelta = thisWeekHours - lastWeekHours;
    final hoursDeltaPct =
        lastWeekHours > 0 ? (hoursDelta / lastWeekHours * 100) : 0.0;

    final thisWeekSessions = thisWeek.length;
    final lastWeekSessions = lastWeek.length;
    final sessionDelta = thisWeekSessions - lastWeekSessions;

    final thisWeekIntensity = thisWeek.isEmpty
        ? 0.0
        : thisWeek
                .map((s) => (s['intensity'] ?? 0.0).toDouble())
                .reduce((a, b) => a + b) /
            thisWeek.length;
    final lastWeekIntensity = lastWeek.isEmpty
        ? 0.0
        : lastWeek
                .map((s) => (s['intensity'] ?? 0.0).toDouble())
                .reduce((a, b) => a + b) /
            lastWeek.length;
    final intensityDelta = thisWeekIntensity - lastWeekIntensity;

    final subjectChanges = _analyzeSubjectChanges(thisWeek, lastWeek);

    String summary;
    List<String> factors;
    List<String> recommendations;

    if (hoursDeltaPct.abs() < 10 && intensityDelta.abs() < 0.1) {
      summary = 'Your study patterns remained consistent this week.';
      factors = [];
      recommendations = ['Try increasing intensity for breakthrough'];
    } else if (hoursDeltaPct > 10 && intensityDelta > 0.05) {
      summary = 'Strong week! You studied more with better focus.';
      factors = [
        'Study time increased by ${hoursDeltaPct.round()}%',
        'Focus intensity up ${(intensityDelta * 100).round()}%',
      ];
      recommendations = [
        'Maintain this momentum',
        'Consider light review before burnout'
      ];
    } else if (hoursDeltaPct < -10 || intensityDelta < -0.1) {
      summary = 'This week showed a decline in study engagement.';
      factors = [
        if (hoursDeltaPct < -10)
          'Study hours dropped ${(-hoursDeltaPct).round()}%',
        if (intensityDelta < -0.1)
          'Focus quality decreased ${(-intensityDelta * 100).round()}%',
      ];
      recommendations = [
        'Identify blockers (exams, social, health)',
        'Set micro-goals for next week',
      ];
    } else {
      summary = 'Mixed week - some areas improved, others need attention.';
      factors = [
        if (hoursDeltaPct > 0) 'More hours invested',
        if (hoursDeltaPct < 0) 'Fewer hours studied',
        if (intensityDelta > 0) 'Better focus quality',
        if (intensityDelta < 0) 'Focus was harder to maintain',
      ];
      recommendations = ['Balance time investment with intensity'];
    }

    return WeeklyInsight(
      summary: summary,
      hoursThisWeek: thisWeekHours,
      hoursLastWeek: lastWeekHours,
      hoursDelta: hoursDelta,
      hoursDeltaPercent: hoursDeltaPct,
      sessionsThisWeek: thisWeekSessions,
      sessionsLastWeek: lastWeekSessions,
      sessionDelta: sessionDelta,
      averageIntensity: thisWeekIntensity,
      intensityDelta: intensityDelta,
      subjectChanges: subjectChanges,
      factors: factors,
      recommendations: recommendations,
      peakDayThisWeek: _getPeakDay(thisWeek),
      peakHourThisWeek: _getPeakHour(thisWeek),
    );
  }

  List<SubjectChange> _analyzeSubjectChanges(
      List<Map<String, dynamic>> thisWeek,
      List<Map<String, dynamic>> lastWeek) {
    final thisWeekBySubject = <String, double>{};
    final lastWeekBySubject = <String, double>{};

    for (final s in thisWeek) {
      final subject = s['subject']?.toString() ?? 'Unknown';
      thisWeekBySubject[subject] =
          (thisWeekBySubject[subject] ?? 0) + ((s['duration'] ?? 0) / 60);
    }

    for (final s in lastWeek) {
      final subject = s['subject']?.toString() ?? 'Unknown';
      lastWeekBySubject[subject] =
          (lastWeekBySubject[subject] ?? 0) + ((s['duration'] ?? 0) / 60);
    }

    final changes = <SubjectChange>[];
    final allSubjects = {...thisWeekBySubject.keys, ...lastWeekBySubject.keys};

    for (final subject in allSubjects) {
      final thisHours = thisWeekBySubject[subject] ?? 0;
      final lastHours = lastWeekBySubject[subject] ?? 0;
      changes.add(SubjectChange(
        subject: subject,
        hoursThisWeek: thisHours,
        hoursLastWeek: lastHours,
        change: thisHours - lastHours,
        changePercent: lastHours > 0
            ? ((thisHours - lastHours) / lastHours * 100)
            : (thisHours > 0 ? 100 : 0),
      ));
    }

    changes.sort((a, b) => b.change.abs().compareTo(a.change.abs()));
    return changes.take(5).toList();
  }

  List<Map<String, dynamic>> _getWeekSessions(
      List<Map<String, dynamic>> sessions, DateTime start,
      [DateTime? end]) {
    final endDate = end ?? DateTime.now();
    return sessions.where((s) {
      final date = _safeParseDate(s['date']);
      return date.isAfter(start) && date.isBefore(endDate);
    }).toList();
  }

  int _getPeakDay(List<Map<String, dynamic>> sessions) {
    final byDay = <int, double>{};
    for (final s in sessions) {
      final date = _safeParseDate(s['date']);
      final day = date.weekday - 1;
      byDay[day] = (byDay[day] ?? 0) + ((s['duration'] ?? 0) / 60);
    }
    if (byDay.isEmpty) return 0;
    return byDay.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  int _getPeakHour(List<Map<String, dynamic>> sessions) {
    final byHour = <int, double>{};
    for (final s in sessions) {
      final date = _safeParseDate(s['date']);
      byHour[date.hour] =
          (byHour[date.hour] ?? 0) + ((s['duration'] ?? 0) / 60);
    }
    if (byHour.isEmpty) return 0;
    return byHour.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  // ─────────────────────────────────────────────────────────────────
  // DATA HELPERS
  // ─────────────────────────────────────────────────────────────────

  Future<List<Map<String, dynamic>>> _getSessionHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_sessionHistoryKey);
    if (raw == null || raw.isEmpty) return [];

    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (_) {
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> _getMockScores() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_mockScoresKey);
    if (raw == null || raw.isEmpty) return [];

    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (_) {
      return [];
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// TIME DISTRIBUTION
// ─────────────────────────────────────────────────────────────────

class SubjectPerformanceData {
  String subject;
  int totalMinutes;
  int sessionCount;
  double totalIntensity;
  double totalFocusQuality;
  List<String> chaptersStudied;
  double averageIntensity = 0;
  double averageFocusQuality = 0;
  double trend = 0;
  double averageMockScore = 0;
  List<String> weakChapters = [];
  List<String> strongChapters = [];

  SubjectPerformanceData({
    required this.subject,
    required this.totalMinutes,
    required this.sessionCount,
    required this.totalIntensity,
    required this.totalFocusQuality,
    required this.chaptersStudied,
  });
}

class SubjectPerformance {
  final String subject;
  final int totalSessions;
  final int totalMinutes;
  final double averageScore;
  final List<DailyPerformance> dailyPerformances;
  final int strongDays;
  final int weakDays;
  final double averageIntensity;
  final double averageFocusQuality;
  final List<String> chaptersStudied;
  final List<String> weakChapters;
  final List<String> strongChapters;
  final double trend;

  SubjectPerformance({
    required this.subject,
    required this.totalSessions,
    required this.totalMinutes,
    required this.averageScore,
    required this.dailyPerformances,
    this.strongDays = 0,
    this.weakDays = 0,
    this.averageIntensity = 0,
    this.averageFocusQuality = 0,
    this.chaptersStudied = const [],
    this.weakChapters = const [],
    this.strongChapters = const [],
    this.trend = 0,
  });

  Map<String, dynamic> toJson() => {
        'subject': subject,
        'totalSessions': totalSessions,
        'totalMinutes': totalMinutes,
        'averageScore': averageScore,
        'dailyPerformances': dailyPerformances.map((d) => d.toJson()).toList(),
        'strongDays': strongDays,
        'weakDays': weakDays,
      };
}

class DailyPerformance {
  final DateTime date;
  final double score;
  final int minutes;

  DailyPerformance({
    required this.date,
    required this.score,
    required this.minutes,
  });

  Map<String, dynamic> toJson() => {
        'date': date.toIso8601String(),
        'score': score,
        'minutes': minutes,
      };

  factory DailyPerformance.fromJson(Map<String, dynamic> json) {
    DateTime date;
    try {
      date = DateTime.parse(json['date'] as String);
    } catch (_) {
      date = DateTime.now();
    }
    return DailyPerformance(
      date: date,
      score: (json['score'] as num).toDouble(),
      minutes: json['minutes'] as int,
    );
  }
}

class CorrelationAnalysis {
  final double correlation;
  final double slope;
  final double intercept;
  final List<WeeklyDataPoint> weeklyData;
  final String insight;
  final bool isSignificant;

  CorrelationAnalysis({
    required this.correlation,
    required this.slope,
    required this.intercept,
    required this.weeklyData,
    required this.insight,
    required this.isSignificant,
  });
}

class WeeklyDataPoint {
  final String week;
  final double studyHours;
  final double mockScore;

  WeeklyDataPoint({
    required this.week,
    required this.studyHours,
    required this.mockScore,
  });
}

class DelayedChapter {
  final String subject;
  final String chapter;
  final int? daysSinceStudied;
  final int studyCount;
  final bool isIgnored;
  final int priority;

  DelayedChapter({
    required this.subject,
    required this.chapter,
    this.daysSinceStudied,
    required this.studyCount,
    required this.isIgnored,
    required this.priority,
  });
}

class DelayedChaptersReport {
  final List<DelayedChapter> delayed;
  final List<DelayedChapter> ignored;
  final int totalIgnored;
  final int totalDelayed;

  const DelayedChaptersReport({
    required this.delayed,
    required this.ignored,
    required this.totalIgnored,
    required this.totalDelayed,
  });
}

class StudyHeatmap {
  final Map<int, double> byDayOfWeek;
  final Map<int, double> byHour;
  final Map<int, Map<int, double>> byDayAndHour;
  final Map<int, Map<int, double>> intensityByDayHour;
  final Map<int, double> normalizedByDayOfWeek;
  final Map<int, double> normalizedByHour;
  final int peakHour;
  final int peakDay;
  final double totalStudyHours;

  StudyHeatmap({
    required this.byDayOfWeek,
    required this.byHour,
    required this.byDayAndHour,
    required this.intensityByDayHour,
    required this.normalizedByDayOfWeek,
    required this.normalizedByHour,
    required this.peakHour,
    required this.peakDay,
    required this.totalStudyHours,
  });
}

class MockDataPoint {
  final DateTime date;
  final double score;
  final String subject;

  MockDataPoint({
    required this.date,
    required this.score,
    required this.subject,
  });
}

class PredictionPoint {
  final DateTime date;
  final double predictedScore;
  final double confidence;

  PredictionPoint({
    required this.date,
    required this.predictedScore,
    required this.confidence,
  });
}

class MockTrendAnalysis {
  final List<double> trendLine;
  final List<double> upperBound;
  final List<double> lowerBound;
  final List<MockDataPoint> dataPoints;
  final double averageScore;
  final double improvement;
  final double confidence;
  final List<PredictionPoint> predictions;

  MockTrendAnalysis({
    required this.trendLine,
    required this.upperBound,
    required this.lowerBound,
    required this.dataPoints,
    required this.averageScore,
    required this.improvement,
    required this.confidence,
    required this.predictions,
  });
}

class SubjectChange {
  final String subject;
  final double hoursThisWeek;
  final double hoursLastWeek;
  final double change;
  final double changePercent;

  SubjectChange({
    required this.subject,
    required this.hoursThisWeek,
    required this.hoursLastWeek,
    required this.change,
    required this.changePercent,
  });
}

class WeeklyInsight {
  final String summary;
  final double hoursThisWeek;
  final double hoursLastWeek;
  final double hoursDelta;
  final double hoursDeltaPercent;
  final int sessionsThisWeek;
  final int sessionsLastWeek;
  final int sessionDelta;
  final double averageIntensity;
  final double intensityDelta;
  final List<SubjectChange> subjectChanges;
  final List<String> factors;
  final List<String> recommendations;
  final int peakDayThisWeek;
  final int peakHourThisWeek;

  WeeklyInsight({
    required this.summary,
    required this.hoursThisWeek,
    required this.hoursLastWeek,
    required this.hoursDelta,
    required this.hoursDeltaPercent,
    required this.sessionsThisWeek,
    required this.sessionsLastWeek,
    required this.sessionDelta,
    required this.averageIntensity,
    required this.intensityDelta,
    required this.subjectChanges,
    required this.factors,
    required this.recommendations,
    required this.peakDayThisWeek,
    required this.peakHourThisWeek,
  });
}

class TimeDistribution {
  final Map<String, double> bySubject;
  final Map<int, double> byHour;
  final Map<int, double> byDayOfWeek;
  final double totalHours;

  TimeDistribution({
    required this.bySubject,
    required this.byHour,
    required this.byDayOfWeek,
    required this.totalHours,
  });
}

// Extension methods for compatibility with existing dashboard
extension AnalyticsServiceExtensions on AnalyticsService {
  Future<Map<String, dynamic>> getOverallStats() async {
    final sessions = await _getSessionHistory();
    final mockScores = await _getMockScores();

    final totalMinutes =
        sessions.fold(0, (sum, s) => sum + ((s['duration'] ?? 0) as int));
    final totalHours = totalMinutes / 60.0;
    final avgIntensity = sessions.isEmpty
        ? 0.0
        : sessions
                .map((s) => (s['intensity'] ?? 0.0) as double)
                .reduce((a, b) => a + b) /
            sessions.length;
    final avgFocus = sessions.isEmpty
        ? 0.0
        : sessions
                .map((s) => (s['focusQuality'] ?? 0.0) as double)
                .reduce((a, b) => a + b) /
            sessions.length;

    final now = DateTime.now();
    final weekSessions = sessions.where((s) {
      final date = _safeParseDate(s['date'], now);
      return date.isAfter(now.subtract(const Duration(days: 7)));
    }).length;

    double latestScore = 0;
    double firstScore = 0;
    double avgMockScore = 0;
    if (mockScores.isNotEmpty) {
      mockScores
          .sort((a, b) => (a['date'] as String).compareTo(b['date'] as String));
      firstScore = (mockScores.first['score'] ?? 0).toDouble();
      latestScore = (mockScores.last['score'] ?? 0).toDouble();
      // Calculate average mock score
      final scoresSum = mockScores.fold(
          0.0, (sum, s) => sum + ((s['score'] ?? 0) as num).toDouble());
      avgMockScore = scoresSum / mockScores.length;
    }

    return {
      'totalHours': totalHours,
      'totalSessions': sessions.length,
      'thisWeekSessions': weekSessions,
      'averageIntensity': avgIntensity,
      'averageFocus': avgFocus,
      'averageScore': avgMockScore, // Fixed: now returns average score
      'latestMockScore': latestScore,
      'scoreImprovement': latestScore - firstScore,
      'subjectsStudied':
          sessions.map((s) => s['subject'] ?? 'Unknown').toSet().length,
    };
  }

  Future<Map<String, SubjectPerformance>> getSubjectPerformances() async {
    final performances = await getSubjectPerformance();
    return {for (var p in performances) p.subject: p};
  }

  Future<TimeDistribution> getTimeDistribution() async {
    final sessions = await _getSessionHistory();

    final bySubject = <String, double>{};
    final byHour = <int, double>{};
    final byDayOfWeek = <int, double>{};
    double totalHours = 0;

    for (final session in sessions) {
      final date = _safeParseDate(session['date']);
      final hours = ((session['duration'] ?? 0) as int) / 60.0;
      final subject = session['subject'] ?? 'Unknown';

      bySubject[subject] = (bySubject[subject] ?? 0) + hours;
      byHour[date.hour] = (byHour[date.hour] ?? 0) + hours;
      byDayOfWeek[date.weekday - 1] =
          (byDayOfWeek[date.weekday - 1] ?? 0) + hours;
      totalHours += hours;
    }

    return TimeDistribution(
      bySubject: bySubject,
      byHour: byHour,
      byDayOfWeek: byDayOfWeek,
      totalHours: totalHours,
    );
  }

  Future<List<String>> identifyWeakAreas() async {
    final report = await getDelayedChapters();
    return [
      ...report.ignored.take(3).map((c) => '${c.subject}: ${c.chapter}'),
      ...report.delayed.take(3).map((c) => '${c.subject}: ${c.chapter}'),
    ];
  }

  Future<List<Map<String, dynamic>>> getPerformanceTrends(
      {int days = 30}) async {
    final mockScores = await _getMockScores();
    final now = DateTime.now();

    final relevant = mockScores.where((s) {
      final date = _safeParseDate(s['date'], now);
      return date.isAfter(now.subtract(Duration(days: days)));
    }).toList()
      ..sort((a, b) => (a['date'] as String).compareTo(b['date'] as String));

    return relevant
        .map((s) => {
              'date': s['date'],
              'score': s['score'],
              'subject': s['subject'],
            })
        .toList();
  }
}

// Provider
final analyticsServiceProvider = AnalyticsService();
