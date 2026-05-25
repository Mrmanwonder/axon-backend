import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/achievement_service.dart';
import '../../services/gamification_service.dart';
import '../../theme/app_theme.dart';

class AchievementGalleryScreen extends ConsumerStatefulWidget {
  const AchievementGalleryScreen({super.key});

  @override
  ConsumerState<AchievementGalleryScreen> createState() =>
      _AchievementGalleryScreenState();
}

class _AchievementGalleryScreenState
    extends ConsumerState<AchievementGalleryScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => popOrPop(context),
        ),
        title: Text(
          'Achievements',
          style: GoogleFonts.googleSans(
            fontWeight: FontWeight.w700,
          ),
        ),
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          tabs: const [
            Tab(text: 'All'),
            Tab(text: 'Streaks'),
            Tab(text: 'Sessions'),
            Tab(text: 'Academic'),
            Tab(text: 'Special'),
          ],
          labelStyle: GoogleFonts.googleSans(fontWeight: FontWeight.w600),
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _AchievementGrid(category: null),
          _AchievementGrid(category: BadgeCategory.streak),
          _AchievementGrid(category: BadgeCategory.session),
          _AchievementGrid(category: BadgeCategory.academic),
          _AchievementGrid(category: BadgeCategory.special),
        ],
      ),
    );
  }

  void popOrPop(BuildContext context) {
    if (Navigator.canPop(context)) {
      Navigator.pop(context);
    } else {
      context.go('/home');
    }
  }
}

class _AchievementGrid extends StatelessWidget {
  final BadgeCategory? category;

  const _AchievementGrid({this.category});

  @override
  Widget build(BuildContext context) {
    final achievements = category == null
        ? AchievementService.allAchievements
        : AchievementService.allAchievements
            .where((a) => a.category == category)
            .toList();

    if (achievements.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.emoji_events_outlined,
                size: 64, color: Colors.amber.withValues(alpha: 0.5)),
            const SizedBox(height: 16),
            Text(
              'No achievements yet',
              style: GoogleFonts.googleSans(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: AxonColors.textSecondary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Keep studying to unlock achievements!',
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
              ),
            ),
          ],
        ),
      );
    }

    return GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 0.85,
      ),
      itemCount: achievements.length,
      itemBuilder: (context, index) {
        return _AchievementCard(achievement: achievements[index]);
      },
    );
  }
}

class _AchievementCard extends StatelessWidget {
  final Achievement achievement;

  const _AchievementCard({required this.achievement});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _showAchievementDetails(context),
      child: Container(
        decoration: BoxDecoration(
          color: AxonColors.surfaceElevated,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: achievement.categoryColor.withValues(alpha: 0.3),
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    achievement.categoryColor.withValues(alpha: 0.3),
                    achievement.categoryColor.withValues(alpha: 0.1),
                  ],
                ),
              ),
              child: Icon(
                achievement.icon,
                color: achievement.categoryColor,
                size: 22,
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Text(
                achievement.name,
                style: GoogleFonts.googleSans(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: AxonColors.textPrimary,
                ),
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      )
          .animate()
          .fadeIn(delay: Duration(milliseconds: 50))
          .scale(begin: const Offset(0.8, 0.8), duration: 200.ms),
    );
  }

  void _showAchievementDetails(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    achievement.categoryColor.withValues(alpha: 0.4),
                    achievement.categoryColor.withValues(alpha: 0.1),
                  ],
                ),
              ),
              child: Icon(
                achievement.icon,
                color: achievement.categoryColor,
                size: 40,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              achievement.name,
              style: GoogleFonts.googleSans(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: AxonColors.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              achievement.description,
              style: GoogleFonts.googleSans(
                fontSize: 14,
                color: AxonColors.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.amber.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.amber.withValues(alpha: 0.3),
                ),
              ),
              child: Text(
                '+${achievement.xpReward} XP',
                style: GoogleFonts.orbitron(
                  color: Colors.amber,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class StatsGamificationScreen extends ConsumerWidget {
  const StatsGamificationScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () {
            if (Navigator.canPop(context)) {
              Navigator.pop(context);
            } else {
              context.go('/home');
            }
          },
        ),
        title: Text(
          'Progress',
          style: GoogleFonts.googleSans(fontWeight: FontWeight.w700),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _StatsCards(),
          ],
        ),
      ),
    );
  }
}

class _StatsCards extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _GamificationOverview(),
        const SizedBox(height: 24),
        Text(
          'Streaks',
          style: GoogleFonts.googleSans(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: AxonColors.textPrimary,
          ),
        ),
        const SizedBox(height: 12),
        const _StreakCard(),
        const SizedBox(height: 24),
        Text(
          'Achievements',
          style: GoogleFonts.googleSans(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: AxonColors.textPrimary,
          ),
        ),
        const SizedBox(height: 12),
        const _AchievementsCard(),
      ],
    );
  }
}

class _GamificationOverview extends StatelessWidget {
  const _GamificationOverview();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: GamificationService.instance.getUserLevel(),
      builder: (context, snapshot) {
        final level = snapshot.data;
        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: AxonGradients.accentGradient,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 60,
                    height: 60,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withValues(alpha: 0.2),
                    ),
                    child: Center(
                      child: Text(
                        '${level?.level ?? 1}',
                        style: GoogleFonts.orbitron(
                          fontSize: 24,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          level?.levelTitle ?? 'Novice Learner',
                          style: GoogleFonts.googleSans(
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Level ${level?.level ?? 1}',
                          style: GoogleFonts.googleSans(
                            fontSize: 14,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: level?.progressToNextLevel ?? 0,
                  minHeight: 8,
                  backgroundColor: Colors.white.withValues(alpha: 0.3),
                  valueColor: const AlwaysStoppedAnimation(Colors.white),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '${level?.currentXp ?? 0} / ${level?.xpToNextLevel ?? 100} XP to next level',
                style: GoogleFonts.googleSans(
                  fontSize: 12,
                  color: Colors.white70,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _StreakCard extends StatelessWidget {
  const _StreakCard();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: GamificationService.instance.getStreak(),
      builder: (context, snapshot) {
        final streak = snapshot.data;
        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.orange.withValues(alpha: 0.3)),
          ),
          child: Row(
            children: [
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: Colors.orange.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(
                  Icons.local_fire_department,
                  color: Colors.orange,
                  size: 32,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${streak?.currentStreak ?? 0} day streak',
                      style: GoogleFonts.googleSans(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: AxonColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Best: ${streak?.longestStreak ?? 0} days',
                      style: GoogleFonts.googleSans(
                        fontSize: 14,
                        color: AxonColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _AchievementsCard extends StatelessWidget {
  const _AchievementsCard();

  @override
  Widget build(BuildContext context) {
    final achievementService = AchievementService.instance;
    return FutureBuilder(
      future: achievementService.getUnlockedAchievements(),
      builder: (context, snapshot) {
        final unlocked = snapshot.data ?? [];
        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.purple.withValues(alpha: 0.3)),
          ),
          child: Row(
            children: [
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: Colors.purple.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(
                  Icons.emoji_events,
                  color: Colors.purple,
                  size: 32,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${unlocked.length} achievements',
                      style: GoogleFonts.googleSans(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: AxonColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${AchievementService.allAchievements.length - unlocked.length} remaining',
                      style: GoogleFonts.googleSans(
                        fontSize: 14,
                        color: AxonColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
