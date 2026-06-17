import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../services/achievement_service.dart' as svc;
import '../theme/app_theme.dart';

// ─── Badge Painters ───────────────────────────────────────────────────────────

class _HexagonClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    final path = Path();
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = math.min(cx, cy);
    for (int i = 0; i < 6; i++) {
      final angle = (math.pi / 180) * (60 * i - 30);
      final x = cx + r * math.cos(angle);
      final y = cy + r * math.sin(angle);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    path.close();
    return path;
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) => false;
}

class _ShieldClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    final w = size.width, h = size.height;
    return Path()
      ..moveTo(w * 0.5, 0)
      ..lineTo(w, h * 0.25)
      ..lineTo(w, h * 0.65)
      ..quadraticBezierTo(w, h * 0.85, w * 0.5, h)
      ..quadraticBezierTo(0, h * 0.85, 0, h * 0.65)
      ..lineTo(0, h * 0.25)
      ..close();
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) => false;
}

// ─── Badge Widget ─────────────────────────────────────────────────────────────

class _AchievementBadge extends StatefulWidget {
  final svc.Achievement achievement;
  final bool unlocked;
  final double size;
  const _AchievementBadge(
      {required this.achievement, this.unlocked = true, this.size = 72});

  @override
  State<_AchievementBadge> createState() => _AchievementBadgeState();
}

class _AchievementBadgeState extends State<_AchievementBadge>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 150));
    _scale = Tween<double>(begin: 1.0, end: 1.05)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOut));
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Widget _buildBody() {
    final a = widget.achievement;
    final sz = widget.size;
    final colors = widget.unlocked
        ? [a.categoryColor, a.categoryColor.withValues(alpha: 0.6)]
        : [AxonColors.surfaceElevated, AxonColors.surfaceHighlight];
    final iconColor = widget.unlocked ? AxonColors.textPrimary : AxonColors.textTertiary;

    final cat = a.category;
    final shape = cat == svc.BadgeCategory.streak
        ? 1
        : cat == svc.BadgeCategory.special
            ? 2
            : cat == svc.BadgeCategory.academic
                ? 3
                : 0;

    Widget inner = Container(
      width: sz,
      height: sz,
      decoration: BoxDecoration(
        gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: colors),
        shape: shape == 0 ? BoxShape.circle : BoxShape.rectangle,
        borderRadius: shape == 3 ? BorderRadius.circular(sz * 0.22) : null,
      ),
      child: Center(child: Icon(a.icon, color: iconColor, size: sz * 0.42)),
    );

    if (shape == 1) {
      inner = ClipPath(
        clipper: _HexagonClipper(),
        child: Container(
          width: sz,
          height: sz,
          decoration: BoxDecoration(
              gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: colors)),
          child: Center(child: Icon(a.icon, color: iconColor, size: sz * 0.42)),
        ),
      );
    } else if (shape == 2) {
      inner = ClipPath(
        clipper: _ShieldClipper(),
        child: Container(
          width: sz,
          height: sz,
          decoration: BoxDecoration(
              gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: colors)),
          child: Center(child: Icon(a.icon, color: iconColor, size: sz * 0.42)),
        ),
      );
    }

    if (!widget.unlocked) {
      inner = ColorFiltered(
          colorFilter:
              const ColorFilter.mode(Color(0xFF808080), BlendMode.saturation),
          child: inner);
    }
    return inner;
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _ctrl.forward(),
      onTapUp: (_) {
        _ctrl.reverse();
        _showDetail(context);
      },
      onTapCancel: () => _ctrl.reverse(),
      child: ScaleTransition(scale: _scale, child: _buildBody()),
    );
  }

  void _showDetail(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => _DetailSheet(
          achievement: widget.achievement, unlocked: widget.unlocked),
    );
  }
}

// ─── Detail Sheet ─────────────────────────────────────────────────────────────

class _DetailSheet extends StatelessWidget {
  final svc.Achievement achievement;
  final bool unlocked;
  const _DetailSheet({required this.achievement, required this.unlocked});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 24),
      decoration: BoxDecoration(
        color: AxonColors.surfaceHighlight,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF3A3A3C), width: 0.5),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _AchievementBadge(
              achievement: achievement, unlocked: unlocked, size: 88),
          const SizedBox(height: 16),
          Text(achievement.name,
              style: TextStyle(
                  color: AxonColors.textPrimary,
                  fontSize: 20,
                  fontWeight: FontWeight.w700),
              textAlign: TextAlign.center),
          const SizedBox(height: 6),
          Text(achievement.description,
              style: TextStyle(color: AxonColors.textSecondary, fontSize: 15),
              textAlign: TextAlign.center),
          const SizedBox(height: 6),
          Text('+${achievement.xpReward} XP',
              style: TextStyle(
                  color: AxonColors.good,
                  fontSize: 13,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          Text(unlocked ? 'Unlocked' : 'Not yet earned',
              style: TextStyle(
                  color: unlocked ? AxonColors.good : AxonColors.textTertiary,
                  fontSize: 13)),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: TextButton(
              onPressed: () => Navigator.pop(context),
              style: TextButton.styleFrom(
                backgroundColor: AxonColors.surfaceElevated,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
              child: Text('Done',
                  style: TextStyle(
                      color: AxonColors.textPrimary,
                      fontWeight: FontWeight.w600,
                      fontSize: 16)),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Summary Bar ──────────────────────────────────────────────────────────────

class _SummaryBar extends StatelessWidget {
  final int total;
  final int unlocked;
  const _SummaryBar({required this.total, required this.unlocked});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: AxonColors.surfaceHighlight,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AxonColors.divider, width: 0.5),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _Stat(label: 'Total', value: '$total'),
          Container(width: 0.5, height: 32, color: AxonColors.divider),
          _Stat(
              label: 'Unlocked',
              value: '$unlocked',
              color: AxonColors.good),
          Container(width: 0.5, height: 32, color: AxonColors.divider),
          _Stat(
              label: 'Remaining',
              value: '${total - unlocked}',
              color: AxonColors.textTertiary),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _Stat(
      {required this.label, required this.value, Color? color})
      : color = color ?? const Color(0xFFFFFFFF); // Will be overridden at runtime
  @override
  Widget build(BuildContext context) {
    return Column(children: [
      Text(value,
          style: TextStyle(
              color: color == const Color(0xFFFFFFFF) ? AxonColors.textPrimary : color,
              fontSize: 22, fontWeight: FontWeight.w700)),
      const SizedBox(height: 2),
      Text(label,
          style: TextStyle(color: AxonColors.textTertiary, fontSize: 12)),
    ]);
  }
}

// ─── Main Screen ──────────────────────────────────────────────────────────────

class AchievementsScreen extends StatefulWidget {
  const AchievementsScreen({super.key});
  @override
  State<AchievementsScreen> createState() => _AchievementsScreenState();
}

class _AchievementsScreenState extends State<AchievementsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final _tabs = ['All', 'Streaks', 'Sessions', 'Academic', 'Special'];
  List<svc.Achievement> _unlocked = [];
  bool _loaded = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: _tabs.length, vsync: this);
    _load();
  }

  Future<void> _load() async {
    _unlocked = await svc.AchievementService.instance.getUnlockedAchievements();
    if (mounted) setState(() => _loaded = true);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  List<svc.Achievement> _filter(int idx) {
    final all = svc.AchievementService.allAchievements;
    if (idx == 0) return all;
    final cat = [
      svc.BadgeCategory.streak,
      svc.BadgeCategory.session,
      svc.BadgeCategory.academic,
      svc.BadgeCategory.special
    ][idx - 1];
    return all
        .where((a) =>
            a.category == cat ||
            (idx == 4 &&
                (a.category == svc.BadgeCategory.special ||
                    a.category == svc.BadgeCategory.social)))
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    final total = svc.AchievementService.allAchievements.length;
    final unlockedCount = _unlocked.length;
    const brandBlue = Color(0xFF3A86FF); // Sleek UI Brand Accent Color

    return Scaffold(
      backgroundColor: AxonColors.background,
      body: NestedScrollView(
        headerSliverBuilder: (context, inner) => [
          SliverAppBar(
            backgroundColor: AxonColors.background,
            pinned: true,
            expandedHeight: 120,
            elevation: 0,
            leading: IconButton(
              icon: const Icon(Icons.arrow_back_ios_new, color: brandBlue, size: 20),
              onPressed: () => Navigator.maybePop(context),
            ),
            flexibleSpace: FlexibleSpaceBar(
              titlePadding: const EdgeInsets.only(left: 20, bottom: 16),
              title: Text('Achievements',
                  style: TextStyle(
                      color: AxonColors.textPrimary,
                      fontSize: 26,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.5)),
            ),
            bottom: PreferredSize(
              preferredSize: const Size.fromHeight(44),
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 16),
                height: 36,
                decoration: BoxDecoration(
                    color: AxonColors.surfaceHighlight,
                    borderRadius: BorderRadius.circular(10)),
                child: TabBar(
                  controller: _tabController,
                  indicator: BoxDecoration(
                      color: AxonColors.surfaceElevated,
                      borderRadius: BorderRadius.circular(8)),
                  indicatorSize: TabBarIndicatorSize.tab,
                  dividerColor: Colors.transparent,
                  labelColor: AxonColors.textPrimary,
                  unselectedLabelColor: AxonColors.textTertiary,
                  labelStyle: const TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w600),
                  tabs: _tabs.map((t) => Tab(text: t)).toList(),
                  onTap: (_) => setState(() {}),
                ),
              ),
            ),
          ),
        ],
        body: _loaded
            ? TabBarView(
                controller: _tabController,
                children: List.generate(_tabs.length, (idx) {
                  final filtered = _filter(idx);
                  return CustomScrollView(
                    key: PageStorageKey<String>(_tabs[idx]),
                    slivers: [
                      // Summary Section Box
                      SliverPadding(
                        padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                        sliver: SliverToBoxAdapter(
                          child: _SummaryBar(total: total, unlocked: unlockedCount),
                        ),
                      ),
                      // Section Header Title
                      SliverPadding(
                        padding: const EdgeInsets.fromLTRB(20, 12, 16, 12),
                        sliver: SliverToBoxAdapter(
                          child: Text(
                            idx == 0 ? 'All Achievements' : _tabs[idx],
                            style: TextStyle(
                                color: AxonColors.textPrimary,
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                                letterSpacing: -0.2),
                          ),
                        ),
                      ),
                      // High-Performance Grid Native Layout
                      SliverPadding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                        sliver: SliverGrid(
                          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3,
                            crossAxisSpacing: 12,
                            mainAxisSpacing: 20,
                            childAspectRatio: 0.75,
                          ),
                          delegate: SliverChildBuilderDelegate(
                            (context, index) {
                              final a = filtered[index];
                              final isUnlocked = _unlocked.any((u) => u.id == a.id);
                              return Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  _AchievementBadge(
                                      achievement: a, unlocked: isUnlocked, size: 72),
                                  const SizedBox(height: 8),
                                  Text(a.name,
                                      textAlign: TextAlign.center,
                                      maxLines: 2,
                                      overflow: TextOverflow.ellipsis,
                                      style: TextStyle(
                                          color: isUnlocked ? AxonColors.textPrimary : AxonColors.textTertiary,
                                          fontSize: 11,
                                          fontWeight: FontWeight.w500,
                                          height: 1.2)),
                                  const SizedBox(height: 2),
                                  Text(isUnlocked ? 'Unlocked' : 'Locked',
                                      style: TextStyle(
                                          color: isUnlocked
                                              ? AxonColors.good
                                              : AxonColors.textTertiary,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w600)),
                                ],
                              );
                            },
                            childCount: filtered.length,
                          ),
                        ),
                      ),
                    ],
                  );
                }),
              )
            : Center(
                child: CircularProgressIndicator(color: AxonColors.electricCyan)),
      ),
    );
  }
}