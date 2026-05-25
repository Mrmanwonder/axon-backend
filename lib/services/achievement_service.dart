import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/material.dart';

enum BadgeCategory {
  streak,
  session,
  academic,
  social,
  special,
}

class Achievement {
  final String id;
  final String name;
  final String description;
  final String iconName;
  final BadgeCategory category;
  final int xpReward;
  final bool isSecret;

  const Achievement({
    required this.id,
    required this.name,
    required this.description,
    required this.iconName,
    required this.category,
    this.xpReward = 50,
    this.isSecret = false,
  });

  IconData get icon {
    switch (iconName) {
      case 'local_fire_department':
        return Icons.local_fire_department;
      case 'school':
        return Icons.school;
      case 'emoji_events':
        return Icons.emoji_events;
      case 'star':
        return Icons.star;
      case 'psychology':
        return Icons.psychology;
      case 'groups':
        return Icons.groups;
      case 'military_tech':
        return Icons.military_tech;
      case 'workspace_premium':
        return Icons.workspace_premium;
      case 'bolt':
        return Icons.bolt;
      case 'timer':
        return Icons.timer;
      case 'menu_book':
        return Icons.menu_book;
      case 'auto_awesome':
        return Icons.auto_awesome;
      case 'trending_up':
        return Icons.trending_up;
      case 'celebration':
        return Icons.celebration;
      case 'whatshot':
        return Icons.whatshot;
      default:
        return Icons.stars;
    }
  }

  Color get categoryColor {
    switch (category) {
      case BadgeCategory.streak:
        return Colors.orange;
      case BadgeCategory.session:
        return Colors.blue;
      case BadgeCategory.academic:
        return Colors.purple;
      case BadgeCategory.social:
        return Colors.green;
      case BadgeCategory.special:
        return Colors.amber;
    }
  }
}

class AchievementService {
  static final AchievementService instance = AchievementService._();

  static const String _achievementsKey = 'achievements_unlocked';
  static const String _progressKey = 'achievements_progress';

  static const List<Achievement> allAchievements = [
    Achievement(
      id: 'first_session',
      name: 'First Steps',
      description: 'Complete your first study session',
      iconName: 'school',
      category: BadgeCategory.session,
      xpReward: 25,
    ),
    Achievement(
      id: 'first_10_sessions',
      name: 'Getting Started',
      description: 'Complete 10 study sessions',
      iconName: 'menu_book',
      category: BadgeCategory.session,
      xpReward: 100,
    ),
    Achievement(
      id: 'century',
      name: 'Century',
      description: 'Complete 100 study sessions',
      iconName: 'workspace_premium',
      category: BadgeCategory.session,
      xpReward: 500,
    ),
    Achievement(
      id: 'week_streak',
      name: 'Week Warrior',
      description: 'Maintain a 7-day study streak',
      iconName: 'local_fire_department',
      category: BadgeCategory.streak,
      xpReward: 150,
    ),
    Achievement(
      id: 'month_streak',
      name: 'Unstoppable',
      description: 'Maintain a 30-day study streak',
      iconName: 'whatshot',
      category: BadgeCategory.streak,
      xpReward: 500,
    ),
    Achievement(
      id: 'quarter_streak',
      name: 'Quarter Master',
      description: 'Maintain a 90-day study streak',
      iconName: 'military_tech',
      category: BadgeCategory.streak,
      xpReward: 1000,
    ),
    Achievement(
      id: 'novice',
      name: 'Novice Learner',
      description: 'Reach Level 5',
      iconName: 'psychology',
      category: BadgeCategory.academic,
      xpReward: 100,
    ),
    Achievement(
      id: 'apprentice',
      name: 'Apprentice',
      description: 'Reach Level 10',
      iconName: 'trending_up',
      category: BadgeCategory.academic,
      xpReward: 250,
    ),
    Achievement(
      id: 'scholar',
      name: 'Scholar',
      description: 'Reach Level 25',
      iconName: 'emoji_events',
      category: BadgeCategory.academic,
      xpReward: 500,
    ),
    Achievement(
      id: 'master',
      name: 'Master',
      description: 'Reach Level 50',
      iconName: 'workspace_premium',
      category: BadgeCategory.academic,
      xpReward: 1000,
    ),
    Achievement(
      id: 'xp_1k',
      name: 'Knowledge Seeker',
      description: 'Earn 1,000 XP',
      iconName: 'bolt',
      category: BadgeCategory.academic,
      xpReward: 100,
    ),
    Achievement(
      id: 'xp_10k',
      name: 'XP Master',
      description: 'Earn 10,000 XP',
      iconName: 'star',
      category: BadgeCategory.academic,
      xpReward: 500,
    ),
    Achievement(
      id: 'first_quiz',
      name: 'Quiz Taker',
      description: 'Complete your first quiz',
      iconName: 'psychology',
      category: BadgeCategory.academic,
      xpReward: 50,
    ),
    Achievement(
      id: 'quiz_master',
      name: 'Quiz Master',
      description: 'Score 100% on 5 quizzes',
      iconName: 'auto_awesome',
      category: BadgeCategory.academic,
      xpReward: 300,
    ),
    Achievement(
      id: 'early_bird',
      name: 'Early Bird',
      description: 'Study before 7 AM',
      iconName: 'timer',
      category: BadgeCategory.special,
      xpReward: 75,
    ),
    Achievement(
      id: 'night_owl',
      name: 'Night Owl',
      description: 'Study after 10 PM',
      iconName: 'timer',
      category: BadgeCategory.special,
      xpReward: 75,
    ),
    Achievement(
      id: 'perfect_week',
      name: 'Perfect Week',
      description: 'Study every day for a week',
      iconName: 'celebration',
      category: BadgeCategory.streak,
      xpReward: 200,
    ),
  ];

  AchievementService._();

  Future<List<Achievement>> getUnlockedAchievements() async {
    final prefs = await SharedPreferences.getInstance();
    final ids = prefs.getStringList(_achievementsKey) ?? [];
    return allAchievements.where((a) => ids.contains(a.id)).toList();
  }

  Future<bool> isAchievementUnlocked(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final ids = prefs.getStringList(_achievementsKey) ?? [];
    return ids.contains(id);
  }

  Future<int> unlockAchievement(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final ids = prefs.getStringList(_achievementsKey) ?? [];

    if (!ids.contains(id)) {
      ids.add(id);
      await prefs.setStringList(_achievementsKey, ids);

      final achievement = allAchievements.firstWhere(
        (a) => a.id == id,
      );
      return achievement.xpReward;
    }
    return 0;
  }

  Future<List<Achievement>> getLockedAchievements() async {
    final prefs = await SharedPreferences.getInstance();
    final ids = prefs.getStringList(_achievementsKey) ?? [];
    return allAchievements
        .where((a) => !ids.contains(a.id) && !a.isSecret)
        .toList();
  }

  Future<List<Achievement>> getSecretAchievements() async {
    return allAchievements.where((a) => a.isSecret).toList();
  }

  Achievement? getAchievementById(String id) {
    try {
      return allAchievements.firstWhere((a) => a.id == id);
    } catch (e) {
      return null;
    }
  }

  Future<Map<String, dynamic>> getProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_progressKey);
    if (data == null) return {};

    try {
      return jsonDecode(data);
    } catch (e) {
      return {};
    }
  }

  Future<void> updateProgress(String achievementId, dynamic value) async {
    final prefs = await SharedPreferences.getInstance();
    final progress = await getProgress();
    progress[achievementId] = value;
    await prefs.setString(_progressKey, jsonEncode(progress));
  }

  Future<int> checkAndUnlock({
    int? totalSessions,
    int? currentStreak,
    int? level,
    int? totalXp,
    int? perfectQuizzes,
  }) async {
    int totalXpEarned = 0;

    if (totalSessions != null) {
      if (totalSessions >= 1) {
        totalXpEarned += await unlockAchievement('first_session');
      }
      if (totalSessions >= 10) {
        totalXpEarned += await unlockAchievement('first_10_sessions');
      }
      if (totalSessions >= 100) {
        totalXpEarned += await unlockAchievement('century');
      }
    }

    if (currentStreak != null) {
      if (currentStreak >= 7) {
        totalXpEarned += await unlockAchievement('week_streak');
      }
      if (currentStreak >= 30) {
        totalXpEarned += await unlockAchievement('month_streak');
      }
      if (currentStreak >= 90) {
        totalXpEarned += await unlockAchievement('quarter_streak');
      }
    }

    if (level != null) {
      if (level >= 5) {
        totalXpEarned += await unlockAchievement('novice');
      }
      if (level >= 10) {
        totalXpEarned += await unlockAchievement('apprentice');
      }
      if (level >= 25) {
        totalXpEarned += await unlockAchievement('scholar');
      }
      if (level >= 50) {
        totalXpEarned += await unlockAchievement('master');
      }
    }

    if (totalXp != null) {
      if (totalXp >= 1000) {
        totalXpEarned += await unlockAchievement('xp_1k');
      }
      if (totalXp >= 10000) {
        totalXpEarned += await unlockAchievement('xp_10k');
      }
    }

    if (perfectQuizzes != null && perfectQuizzes >= 5) {
      totalXpEarned += await unlockAchievement('quiz_master');
    }

    return totalXpEarned;
  }
}
