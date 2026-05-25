import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class UserLevel {
  final int level;
  final int currentXp;
  final int xpToNextLevel;
  final int totalXp;

  UserLevel({
    required this.level,
    required this.currentXp,
    required this.xpToNextLevel,
    required this.totalXp,
  });

  double get progressToNextLevel =>
      xpToNextLevel > 0 ? currentXp / xpToNextLevel : 1.0;

  String get levelTitle => _getLevelTitle(level);

  static String _getLevelTitle(int level) {
    if (level < 5) return 'Novice Learner';
    if (level < 10) return 'Apprentice';
    if (level < 20) return 'Student';
    if (level < 30) return 'Scholar';
    if (level < 50) return 'Expert';
    if (level < 75) return 'Master';
    return 'Grandmaster';
  }

  Map<String, dynamic> toJson() => {
        'level': level,
        'currentXp': currentXp,
        'xpToNextLevel': xpToNextLevel,
        'totalXp': totalXp,
      };

  factory UserLevel.fromJson(Map<String, dynamic> json) {
    return UserLevel(
      level: json['level'] ?? 1,
      currentXp: json['currentXp'] ?? 0,
      xpToNextLevel: json['xpToNextLevel'] ?? 100,
      totalXp: json['totalXp'] ?? 0,
    );
  }
}

class StudyStreak {
  final int currentStreak;
  final int longestStreak;
  final DateTime? lastStudyDate;
  final List<DateTime> studyDates;

  StudyStreak({
    this.currentStreak = 0,
    this.longestStreak = 0,
    this.lastStudyDate,
    this.studyDates = const [],
  });

  bool get isActiveToday {
    if (lastStudyDate == null) return false;
    final now = DateTime.now();
    return lastStudyDate!.year == now.year &&
        lastStudyDate!.month == now.month &&
        lastStudyDate!.day == now.day;
  }

  bool get canContinueStreak {
    if (lastStudyDate == null) return true;
    final now = DateTime.now();
    final yesterday = now.subtract(const Duration(days: 1));
    final isYesterday = lastStudyDate!.year == yesterday.year &&
        lastStudyDate!.month == yesterday.month &&
        lastStudyDate!.day == yesterday.day;
    if (isYesterday) return true;
    return isInRecoveryWindow;
  }

  bool get isInRecoveryWindow {
    if (lastStudyDate == null) return false;
    final now = DateTime.now();
    final twoDaysAgo = now.subtract(const Duration(days: 2));
    final matchesTwoDaysAgo = lastStudyDate!.year == twoDaysAgo.year &&
        lastStudyDate!.month == twoDaysAgo.month &&
        lastStudyDate!.day == twoDaysAgo.day;
    return matchesTwoDaysAgo && now.hour < 12;
  }

  Duration get recoveryWindowRemaining {
    if (!isInRecoveryWindow) return Duration.zero;
    final now = DateTime.now();
    final deadline = DateTime(now.year, now.month, now.day, 12);
    return deadline.difference(now);
  }

  Map<String, dynamic> toJson() => {
        'currentStreak': currentStreak,
        'longestStreak': longestStreak,
        'lastStudyDate': lastStudyDate?.toIso8601String(),
        'studyDates': studyDates.map((d) => d.toIso8601String()).toList(),
      };

  factory StudyStreak.fromJson(Map<String, dynamic> json) {
    return StudyStreak(
      currentStreak: json['currentStreak'] ?? 0,
      longestStreak: json['longestStreak'] ?? 0,
      lastStudyDate: json['lastStudyDate'] != null
          ? DateTime.tryParse(json['lastStudyDate'])
          : null,
      studyDates: (json['studyDates'] as List?)
              ?.map((d) => DateTime.tryParse(d.toString()) ?? DateTime.now())
              .toList() ??
          [],
    );
  }
}

class GamificationService {
  static final GamificationService instance = GamificationService._();

  static const String _levelKey = 'gamification_level';
  static const String _streakKey = 'gamification_streak';
  static const String _badgesKey = 'gamification_badges';
  static const String _dailyChallengesKey = 'gamification_daily_challenges';

  static int xpForLevel(int level) => (level * 100 * (1 + level * 0.1)).round();

  GamificationService._();

  Future<UserLevel> getUserLevel() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_levelKey);
    if (data == null) {
      return UserLevel(
          level: 1, currentXp: 0, xpToNextLevel: xpForLevel(1), totalXp: 0);
    }

    try {
      return UserLevel.fromJson(jsonDecode(data));
    } catch (e) {
      return UserLevel(
          level: 1, currentXp: 0, xpToNextLevel: xpForLevel(1), totalXp: 0);
    }
  }

  Future<UserLevel> addXp(int xp, {String? reason}) async {
    final level = await getUserLevel();
    final newTotalXp = level.totalXp + xp;
    var newLevel = level.level;
    var currentXp = level.currentXp + xp;
    var xpToNext = xpForLevel(newLevel);

    while (currentXp >= xpToNext) {
      currentXp -= xpToNext;
      newLevel++;
      xpToNext = xpForLevel(newLevel);
    }

    final newUserLevel = UserLevel(
      level: newLevel,
      currentXp: currentXp,
      xpToNextLevel: xpToNext,
      totalXp: newTotalXp,
    );

    await _saveLevel(newUserLevel);
    return newUserLevel;
  }

  Future<void> _saveLevel(UserLevel level) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_levelKey, jsonEncode(level.toJson()));
  }

  int calculateSessionXp(int minutes, double intensity) {
    final baseXp = (minutes * 0.5).round();
    final intensityBonus = (intensity * baseXp).round();
    return baseXp + intensityBonus;
  }

  Future<StudyStreak> getStreak() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_streakKey);
    if (data == null) return StudyStreak();

    try {
      return StudyStreak.fromJson(jsonDecode(data));
    } catch (e) {
      return StudyStreak();
    }
  }

  Future<StudyStreak> logStudySession(int minutes) async {
    var streak = await getStreak();
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);

    final newStudyDates = List<DateTime>.from(streak.studyDates);
    if (!newStudyDates.any((d) =>
        d.year == today.year && d.month == today.month && d.day == today.day)) {
      newStudyDates.add(today);
    }

    int newStreak;
    if (streak.lastStudyDate == null) {
      newStreak = 1;
    } else if (streak.isActiveToday) {
      newStreak = streak.currentStreak;
    } else if (streak.canContinueStreak) {
      newStreak = streak.currentStreak + 1;
    } else {
      newStreak = 1;
    }

    final newLongest =
        newStreak > streak.longestStreak ? newStreak : streak.longestStreak;

    streak = StudyStreak(
      currentStreak: newStreak,
      longestStreak: newLongest,
      lastStudyDate: now,
      studyDates: newStudyDates,
    );

    await _saveStreak(streak);
    return streak;
  }

  Future<void> _saveStreak(StudyStreak streak) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_streakKey, jsonEncode(streak.toJson()));
  }

  Future<List<String>> getUnlockedBadges() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getStringList(_badgesKey);
    return data ?? [];
  }

  Future<void> unlockBadge(String badgeId) async {
    final badges = await getUnlockedBadges();
    if (!badges.contains(badgeId)) {
      badges.add(badgeId);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList(_badgesKey, badges);
    }
  }

  Future<bool> hasBadge(String badgeId) async {
    final badges = await getUnlockedBadges();
    return badges.contains(badgeId);
  }

  Future<List<String>> checkAndUnlockBadges() async {
    final newlyUnlocked = <String>[];
    final level = await getUserLevel();
    final streak = await getStreak();
    final analytics = await _getAnalyticsForBadges();

    if (level.level >= 5 && !await hasBadge('novice')) {
      newlyUnlocked.add('novice');
    }
    if (level.level >= 10 && !await hasBadge('apprentice')) {
      newlyUnlocked.add('apprentice');
    }
    if (level.level >= 25 && !await hasBadge('scholar')) {
      newlyUnlocked.add('scholar');
    }
    if (streak.currentStreak >= 7 && !await hasBadge('week_streak')) {
      newlyUnlocked.add('week_streak');
    }
    if (streak.currentStreak >= 30 && !await hasBadge('month_streak')) {
      newlyUnlocked.add('month_streak');
    }
    if (analytics['totalSessions'] >= 10 &&
        !await hasBadge('first_10_sessions')) {
      newlyUnlocked.add('first_10_sessions');
    }
    if (analytics['totalSessions'] >= 100 && !await hasBadge('century')) {
      newlyUnlocked.add('century');
    }
    if (level.totalXp >= 1000 && !await hasBadge('xp_1k')) {
      newlyUnlocked.add('xp_1k');
    }
    if (level.totalXp >= 10000 && !await hasBadge('xp_10k')) {
      newlyUnlocked.add('xp_10k');
    }

    for (final badge in newlyUnlocked) {
      await unlockBadge(badge);
    }

    return newlyUnlocked;
  }

  Future<Map<String, dynamic>> _getAnalyticsForBadges() async {
    return {
      'totalSessions': 0,
    };
  }

  Future<int> getDailyChallengeXp() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_dailyChallengesKey);
    if (data == null) return 0;

    try {
      final decoded = jsonDecode(data);
      final lastCompleted = DateTime.tryParse(decoded['lastCompleted'] ?? '');
      if (lastCompleted == null) return 0;

      final now = DateTime.now();
      if (lastCompleted.year == now.year &&
          lastCompleted.month == now.month &&
          lastCompleted.day == now.day) {
        return decoded['xpEarned'] ?? 0;
      }
    } catch (e) {
      // Ignore malformed cached daily challenge data.
    }
    return 0;
  }

  Future<void> saveDailyChallengeXp(int xp) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _dailyChallengesKey,
        jsonEncode({
          'lastCompleted': DateTime.now().toIso8601String(),
          'xpEarned': xp,
        }));
  }
}
