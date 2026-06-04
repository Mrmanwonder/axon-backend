import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/models.dart';
import '../components/floating_glass_navbar.dart';
import '../services/app_state.dart';
import '../services/chat_history_service.dart';
import '../services/haptics_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/ask_axon_widget.dart';
import '../screens/study/locked_app_screen.dart';

/// Centralized navigation data for the study workspace.
const List<_NavDestinationData> _kDestinations = [
  _NavDestinationData(
    route: '/home',
    label: 'Dashboard',
    icon: Icons.grid_view_rounded,
    title: 'Study Command Center',
    subtitle: 'Track momentum, deadlines, and the next action.',
  ),
  _NavDestinationData(
    route: '/analysis',
    label: 'Analysis',
    icon: Icons.analytics_outlined,
    title: 'Performance Analysis',
    subtitle: 'Inspect trends, weak spots, and exam readiness.',
  ),

  _NavDestinationData(
    route: '/study',
    label: 'Study',
    icon: Icons.auto_stories_outlined,
    title: 'Study Workspace',
    subtitle: 'Open chapters, drills, and session tools.',
  ),
  _NavDestinationData(
    route: '/settings',
    label: 'Settings',
    icon: Icons.settings_outlined,
    title: 'Workspace Settings',
    subtitle: 'Control sync, profile, and desktop preferences.',
  ),
];

class ShellScreen extends ConsumerStatefulWidget {
  final Widget child;
  const ShellScreen({super.key, required this.child});

  @override
  ConsumerState<ShellScreen> createState() => _ShellScreenState();
}

class _ShellScreenState extends ConsumerState<ShellScreen>
    with TickerProviderStateMixin {
  late final AnimationController _axonAnimController;
  late final AnimationController _flipAnimController;
  late final Animation<double> _axonScaleAnim;
  late final Animation<double> _axonExpandAnim;

  bool _showAxonBar = false;
  String? _lastLocation;

  @override
  void initState() {
    super.initState();
    _axonAnimController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );

    _axonScaleAnim = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _axonAnimController, curve: Curves.easeInOut),
    );

    _axonExpandAnim = CurvedAnimation(
      parent: _axonAnimController,
      curve: Curves.easeInOut,
    );

    _flipAnimController = AnimationController(
      duration: const Duration(milliseconds: 350),
      vsync: this,
    )..addListener(() {
      ref.read(examPlannerFlipProvider.notifier).state = _flipAnimController.value;
    });

    _initializeServices();
  }

  void _initializeServices() {
    Future.microtask(() {
      ChatHistoryService.instance.initialize();
      final user = ref.read(authStateProvider).user;
      if (user != null) {
        ref.read(metricsProvider.notifier).syncWithProfile(user);
      }
      if (mounted) _updateFlipFromRoute(context);
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    WidgetsBinding.instance.addPostFrameCallback((_) => _updateFlipFromRoute(context));
  }

  void _updateFlipFromRoute(BuildContext context) {
    final loc = GoRouterState.of(context).matchedLocation;
    if (loc == _lastLocation) return;
    _lastLocation = loc;

    final isExamRoute = loc.startsWith('/exam/');
    if (isExamRoute) {
      if (!_flipAnimController.isCompleted) _flipAnimController.forward();
    } else {
      if (_flipAnimController.value > 0) _flipAnimController.reverse();
    }
  }

  @override
  void dispose() {
    _axonAnimController.dispose();
    _flipAnimController.dispose();
    super.dispose();
  }

  int _currentIndex(BuildContext context) {
    final loc = GoRouterState.of(context).matchedLocation;
    return _kDestinations.indexWhere((d) => loc.startsWith(d.route)).clamp(0, _kDestinations.length - 1);
  }

  bool _isDesktop(BuildContext context) => MediaQuery.of(context).size.width >= 1120;

  @override
  Widget build(BuildContext context) {
    ref.listen(metricsProvider.select((m) => m.streak), (prev, next) {
      if (next > (prev ?? 0)) AxonHaptics.success();
    });

    ref.listen(blockedAppProvider, (prev, next) {
      if (next != null && prev == null) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          Navigator.of(context).push(
            MaterialPageRoute(
              fullscreenDialog: true,
              builder: (_) => LockedAppScreen(
                appName: next.appName,
                packageName: next.packageName,
              ),
            ),
          ).then((_) {
            BlockedAppOverlay.dismiss(ref);
          });
        });
      }
    });

    final idx = _currentIndex(context);
    final isDark = AxonThemeMode.isDark;

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle(
        systemNavigationBarColor: Colors.transparent,
        systemNavigationBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
        statusBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
      ),
      child: Scaffold(
        backgroundColor: isDark ? const Color(0xFF05070B) : const Color(0xFFF4F7FB),
        extendBody: true,
        body: _isDesktop(context)
            ? _buildDesktopShell(idx)
            : _buildMobileShell(idx),
      ),
    );
  }

  // --- Shell Builders ---

  Widget _buildDesktopShell(int idx) {
    final metrics = ref.watch(metricsProvider);
    final user = ref.watch(authStateProvider).user;

    return Stack(
      children: [
        const _DesktopBackground(),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                _DesktopSidebar(
                  currentIndex: idx,
                  metrics: metrics,
                  onOpenChat: _openAxonChat,
                  onNewChat: _startNewChat,
                ),
                const SizedBox(width: 20),
                Expanded(
                  child: Column(
                    children: [
                      _DesktopHeader(
                        destination: _kDestinations[idx],
                        user: user,
                        metrics: metrics,
                      ),
                      const SizedBox(height: 20),
                      Expanded(
                        child: _ContentWrapper(child: widget.child),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        if (_showAxonBar)
          Positioned(
            bottom: 32,
            right: 32,
            width: 420,
            child: _buildAxonInputBarContent(),
          ),
        _buildStreakOverlay(),
      ],
    );
  }

  Widget _buildMobileShell(int idx) {
    final isNavbarVisible = ref.watch(navbarVisibleProvider);
    const double navHeight = 80;

    return Stack(
      children: [
        Padding(
          padding: EdgeInsets.only(bottom: isNavbarVisible ? navHeight : 0),
          child: widget.child,
        ),
        if (isNavbarVisible)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: FloatingGlassNavbar(currentIndex: idx),
          ),
        if (!_showAxonBar && idx == 1)
          Positioned(
            bottom: isNavbarVisible ? 90 + navHeight : 90,
            right: 20,
            child: _FloatingAxonButton(onTap: _openAxonChat, animation: _axonScaleAnim),
          ),
        if (_showAxonBar)
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: EdgeInsets.fromLTRB(
                20, 0, 20, isNavbarVisible ? 92 + navHeight : 92,
              ),
              child: _buildAxonInputBarContent(),
            ),
          ),
        _buildStreakOverlay(),
      ],
    );
  }

  // --- AI Logic ---

  Widget _buildAxonInputBarContent() {
    return AnimatedBuilder(
      animation: _axonAnimController,
      builder: (context, _) => Opacity(
        opacity: _axonExpandAnim.value,
        child: Transform.scale(
          scale: 0.9 + (_axonExpandAnim.value * 0.1),
          child: AskAxonWidget(
            autofocus: true,
            onSubmit: _startNewChatWithPrompt,
            onPlan: () => _startNewChatWithPrompt('Create a study plan for today.'),
            onDoubt: () => _startNewChatWithPrompt('Help me solve a doubt.'),
          ),
        ),
      ),
    );
  }

  void _openAxonChat() async {
    await ChatHistoryService.instance.initialize();
    if (mounted) context.push('/ai');
  }

  void _closeAxonBar() => _axonAnimController.reverse().then((_) {
    if (mounted) setState(() => _showAxonBar = false);
  });

  void _startNewChat() async {
    await ChatHistoryService.instance.startNewChat();
    _closeAxonBar();
    await Future.delayed(const Duration(milliseconds: 300));
    if (mounted) context.push('/ai');
  }

  void _startNewChatWithPrompt(String prompt) async {
    await ChatHistoryService.instance.startNewChat();
    _closeAxonBar();
    await Future.delayed(const Duration(milliseconds: 300));
    if (mounted) {
      context.push('/ai', extra: {'initialQuestion': prompt});
    }
  }

  // --- Overlays ---

  Widget _buildStreakOverlay() {
    final metrics = ref.watch(metricsProvider);
    if (!metrics.showStreakHighlight) return const SizedBox.shrink();

    return _StreakHighlight(
      streak: metrics.streak,
      onDismiss: () => ref.read(metricsProvider.notifier).setShowStreakHighlight(false),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// SUPPORTING COMPONENTS
// ─────────────────────────────────────────────────────────────────

class _NavDestinationData {
  final String route;
  final String label;
  final IconData icon;
  final String title;
  final String subtitle;

  const _NavDestinationData({
    required this.route,
    required this.label,
    required this.icon,
    required this.title,
    required this.subtitle,
  });
}

/// Reusable glassmorphism card.
class GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;

  const GlassCard({super.key, required this.child, this.padding});

  @override
  Widget build(BuildContext context) => ClipRRect(
    borderRadius: BorderRadius.circular(28),
    child: BackdropFilter(
      filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
      child: Container(
        padding: padding,
        decoration: BoxDecoration(
          color: SpatialColors.charcoalLight.withValues(alpha: 0.86),
          borderRadius: BorderRadius.circular(28),
          border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          boxShadow: SpatialGlow.glassDock,
        ),
        child: child,
      ),
    ),
  );
}

// ── Background ─────────────────────────────────────────────────

class _DesktopBackground extends StatelessWidget {
  const _DesktopBackground();

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF07111F), Color(0xFF0A0F17), Color(0xFF111827)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        Positioned(
          top: -120, right: -40,
          child: _GlowOrb(color: const Color(0xFF3A86FF).withValues(alpha: 0.1)),
        ),
        Positioned(
          bottom: -180, left: -100,
          child: _GlowOrb(color: Colors.white.withValues(alpha: 0.05)),
        ),
      ],
    );
  }
}

class _GlowOrb extends StatelessWidget {
  final Color color;
  const _GlowOrb({required this.color});

  @override
  Widget build(BuildContext context) => Container(
    width: 420,
    height: 420,
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      color: color,
    ),
  );
}

class _ContentWrapper extends StatelessWidget {
  final Widget child;
  const _ContentWrapper({required this.child});

  @override
  Widget build(BuildContext context) => ClipRRect(
    borderRadius: BorderRadius.circular(28),
    child: Container(
      decoration: BoxDecoration(
        color: const Color(0xFF0D1320).withValues(alpha: 0.86),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.26),
            blurRadius: 32,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: child,
    ),
  );
}

// ── Desktop Sidebar ────────────────────────────────────────────

class _DesktopSidebar extends StatelessWidget {
  final int currentIndex;
  final MetricsState metrics;
  final VoidCallback onOpenChat;
  final VoidCallback onNewChat;

  const _DesktopSidebar({
    required this.currentIndex,
    required this.metrics,
    required this.onOpenChat,
    required this.onNewChat,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 284,
      child: GlassCard(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _BrandLockup(),
            const SizedBox(height: 24),
            ...List.generate(
              _kDestinations.length,
              (i) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _DesktopNavItem(
                  data: _kDestinations[i],
                  selected: i == currentIndex,
                  onTap: () {
                    AxonHaptics.mediumImpact();
                    context.go(_kDestinations[i].route);
                  },
                ),
              ),
            ),
            const SizedBox(height: 24),
            _QuickActionCard(
              metrics: metrics,
              onContinue: onOpenChat,
              onNew: onNewChat,
            ),
            const Spacer(),
            _SessionSnapshot(metrics: metrics),
          ],
        ),
      ),
    );
  }
}

class _BrandLockup extends StatelessWidget {
  const _BrandLockup();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 48, height: 48,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              gradient: const LinearGradient(
                colors: [Color(0xFF3A86FF), Color(0xFF0EA5E9)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: const Icon(Icons.bolt_rounded, color: Colors.white, size: 24),
          ),
          const SizedBox(height: 14),
          Text(
            'AXON DESKTOP',
            style: GoogleFonts.robotoMono(
              color: Colors.white, fontSize: 14,
              fontWeight: FontWeight.w700, letterSpacing: 1.4,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Focused study operations for large-screen work.',
            style: GoogleFonts.googleSans(color: Colors.white70, fontSize: 13),
          ),
        ],
      ),
    );
  }
}

class _DesktopNavItem extends StatelessWidget {
  final _NavDestinationData data;
  final bool selected;
  final VoidCallback onTap;

  const _DesktopNavItem({
    required this.data,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: selected
              ? const Color(0xFF3A86FF).withValues(alpha: 0.18)
              : Colors.white.withValues(alpha: 0.03),
          border: Border.all(
            color: selected
                ? const Color(0xFF3A86FF).withValues(alpha: 0.24)
                : Colors.white.withValues(alpha: 0.05),
          ),
        ),
        child: Row(
          children: [
            Icon(
              data.icon,
              color: selected ? Colors.white : Colors.white54,
              size: 20,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                data.label,
                style: GoogleFonts.googleSans(
                  color: selected ? Colors.white : Colors.white70,
                  fontSize: 14,
                  fontWeight: selected ? FontWeight.w600 : FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _QuickActionCard extends StatelessWidget {
  final MetricsState metrics;
  final VoidCallback onContinue;
  final VoidCallback onNew;

  const _QuickActionCard({
    required this.metrics,
    required this.onContinue,
    required this.onNew,
  });

  @override
  Widget build(BuildContext context) {
    final hasPrevious = ChatHistoryService.instance.hasPreviousChat;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF3A86FF).withValues(alpha: 0.20),
            const Color(0xFF0F172A).withValues(alpha: 0.80),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF3A86FF).withValues(alpha: 0.20)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Ask Axon',
            style: GoogleFonts.googleSans(
              color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            hasPrevious
                ? 'Jump back into your last conversation or start a clean thread.'
                : 'Use AI inside the desktop workflow without leaving the workspace.',
            style: GoogleFonts.googleSans(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _OptionButton(
                  icon: Icons.chat_rounded,
                  label: hasPrevious ? 'Continue' : 'Open',
                  onTap: onContinue,
                ),
              ),
              if (hasPrevious) ...[
                const SizedBox(width: 10),
                Expanded(
                  child: _OptionButton(
                    icon: Icons.add_rounded,
                    label: 'New Chat',
                    onTap: onNew,
                    isPrimary: true,
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _MetricBadge(label: 'Streak', value: '${metrics.streak}d'),
              const SizedBox(width: 8),
              _MetricBadge(
                label: 'Focus',
                value: '${(metrics.predictedPerformance * 100).clamp(0, 100).round()}%',
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SessionSnapshot extends StatelessWidget {
  final MetricsState metrics;
  const _SessionSnapshot({required this.metrics});

  @override
  Widget build(BuildContext context) {
    final studyHours = metrics.activeStudyHours.toStringAsFixed(1);
    final screenHours = metrics.screenTimeHours.toStringAsFixed(1);
    final sleepHours = metrics.sleepHours.toStringAsFixed(1);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Session Snapshot',
            style: GoogleFonts.googleSans(
              color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 14),
          _InfoRow(label: 'Study', value: '$studyHours h'),
          _InfoRow(label: 'Screen', value: '$screenHours h'),
          _InfoRow(label: 'Sleep', value: '$sleepHours h'),
        ],
      ),
    );
  }
}

// ── Desktop Header ─────────────────────────────────────────────

class _DesktopHeader extends StatelessWidget {
  final _NavDestinationData destination;
  final UserProfile? user;
  final MetricsState metrics;

  const _DesktopHeader({
    required this.destination,
    this.user,
    required this.metrics,
  });

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  destination.title,
                  style: GoogleFonts.googleSans(
                    color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  destination.subtitle,
                  style: GoogleFonts.googleSans(color: Colors.white70, fontSize: 14),
                ),
              ],
            ),
          ),
          _UserStatusBadge(user: user, performance: metrics.predictedPerformance),
        ],
      ),
    );
  }
}

class _UserStatusBadge extends StatelessWidget {
  final UserProfile? user;
  final double performance;

  const _UserStatusBadge({this.user, required this.performance});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              color: const Color(0xFF3A86FF).withValues(alpha: 0.20),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.person_outline_rounded, color: Colors.white),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                user?.displayName ?? 'Student',
                style: GoogleFonts.googleSans(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
              ),
              Text(
                'Ready ${(performance * 100).clamp(0, 100).round()}%',
                style: GoogleFonts.robotoMono(color: Colors.white60, fontSize: 11),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ── Floating AI Button ─────────────────────────────────────────

class _FloatingAxonButton extends StatelessWidget {
  final VoidCallback onTap;
  final Animation<double> animation;

  const _FloatingAxonButton({required this.onTap, required this.animation});

  @override
  Widget build(BuildContext context) => ScaleTransition(
    scale: animation,
    child: FadeTransition(
      opacity: animation,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: 56, height: 56,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF3A86FF), Color(0xFF0EA5E9)],
            ),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.35),
                blurRadius: 16, offset: const Offset(0, 4),
              ),
            ],
          ),
          child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 28),
        ),
      ),
    ),
  );
}

// ── Streak Highlight Overlay ───────────────────────────────────

class _StreakHighlight extends StatelessWidget {
  final int streak;
  final VoidCallback onDismiss;

  const _StreakHighlight({required this.streak, required this.onDismiss});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onDismiss,
    behavior: HitTestBehavior.opaque,
    child: Container(
      color: Colors.black.withValues(alpha: 0.4),
      child: Center(
        child: GlassCard(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('⚡', style: TextStyle(fontSize: 48)),
              const SizedBox(height: 16),
              Text(
                '$streak DAY STREAK',
                style: GoogleFonts.robotoMono(
                  color: Colors.white, fontSize: 20,
                  letterSpacing: 4, fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'SYSTEM_STABILITY: OPTIMAL',
                style: GoogleFonts.robotoMono(color: const Color(0xFF39D353), fontSize: 10),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}

// ── Shared Small Widgets ───────────────────────────────────────

class _MetricBadge extends StatelessWidget {
  final String label;
  final String value;

  const _MetricBadge({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: GoogleFonts.robotoMono(color: Colors.white54, fontSize: 10),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: GoogleFonts.googleSans(
                color: Colors.white, fontSize: 15, fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _InfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Text(
            label,
            style: GoogleFonts.googleSans(color: Colors.white54, fontSize: 13),
          ),
          const Spacer(),
          Text(
            value,
            style: GoogleFonts.robotoMono(
              color: Colors.white, fontSize: 12, fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _OptionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool isPrimary;

  const _OptionButton({
    required this.icon,
    required this.label,
    required this.onTap,
    this.isPrimary = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: isPrimary
              ? const Color(0xFF3A86FF)
              : Colors.white.withValues(alpha: 0.02),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 18),
            const SizedBox(width: 6),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
