import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/study_lock_service.dart';
import '../../theme/app_theme.dart';

class LockedAppScreen extends ConsumerStatefulWidget {
  final String appName;
  final String packageName;

  const LockedAppScreen({
    super.key,
    required this.appName,
    required this.packageName,
  });

  @override
  ConsumerState<LockedAppScreen> createState() => _LockedAppScreenState();
}

class _LockedAppScreenState extends ConsumerState<LockedAppScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _orbitController;
  late AnimationController _shimmerController;
  late AnimationController _entryController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);

    _orbitController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 4000),
    )..repeat();

    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();

    _entryController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..forward();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _orbitController.dispose();
    _shimmerController.dispose();
    _entryController.dispose();
    super.dispose();
  }

  void _dismiss() {
    if (mounted) {
      BlockedAppOverlay.dismiss(ref);
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _dismiss();
      },
      child: Scaffold(
        backgroundColor: AxonColors.oxfordBlueDark,
        body: Stack(
          children: [
            // Glow orb (exam planner style)
            Positioned(
              top: -120,
              right: -60,
              child: Container(
                width: 320,
                height: 320,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: AxonColors.accent.withValues(alpha: 0.12),
                ),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
                  child: Container(),
                ),
              ),
            ),
            SafeArea(
              child: FadeTransition(
                opacity: _entryController,
                child: ScaleTransition(
                  scale: Tween<double>(begin: 0.92, end: 1.0).animate(
                    CurvedAnimation(
                      parent: _entryController,
                      curve: Curves.easeOutCubic,
                    ),
                  ),
                  child: _buildContent(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent() {
    return Column(
      children: [
        // Header with back button
        Padding(
          padding: const EdgeInsets.fromLTRB(8, 8, 16, 0),
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back_ios_new_rounded,
                    color: Colors.white, size: 20),
                onPressed: _dismiss,
              ),
              const Spacer(),
              Text(
                'FOCUS LOCK',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2,
                ),
              ),
              const SizedBox(width: 48),
            ],
          ),
        ),
        const Spacer(flex: 1),

        // Lock icon with glow
        _AnimatedLockIcon(
          pulseController: _pulseController,
          shimmerController: _shimmerController,
        ).animate().fadeIn(duration: 600.ms).scale(
              begin: const Offset(0.6, 0.6),
              end: const Offset(1, 1),
              duration: 600.ms,
              curve: Curves.easeOutBack,
            ),

        const SizedBox(height: 32),

        // Title
        Text(
          'Stay Focused',
          style: GoogleFonts.googleSans(
            fontSize: 28,
            fontWeight: FontWeight.w800,
            color: Colors.white,
            letterSpacing: 0,
          ),
        ).animate().fadeIn(delay: 200.ms, duration: 500.ms).slideY(
              begin: 0.1,
              end: 0,
              delay: 200.ms,
              duration: 500.ms,
            ),

        const SizedBox(height: 12),

        // Subtitle
        Column(
          children: [
            Text(
              '${widget.appName} is locked while you',
              style: GoogleFonts.googleSans(
                fontSize: 14,
                color: AxonColors.textSecondary,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              'complete your study session',
              style: GoogleFonts.googleSans(
                fontSize: 14,
                color: AxonColors.textSecondary,
              ),
            ),
          ],
        ).animate().fadeIn(delay: 400.ms, duration: 500.ms),

        const SizedBox(height: 40),

        // Glass card with progress
        _ProgressCard(
          pulseController: _pulseController,
          orbitController: _orbitController,
        ).animate().fadeIn(delay: 300.ms, duration: 600.ms).slideY(
              begin: 0.15,
              end: 0,
              delay: 300.ms,
              duration: 600.ms,
              curve: Curves.easeOutCubic,
            ),

        const SizedBox(height: 32),

        // Stats row
        _StatsRow().animate().fadeIn(delay: 500.ms, duration: 500.ms),

        const Spacer(flex: 2),

        // Dismiss hint
        GestureDetector(
          onTap: _dismiss,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.1),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.keyboard_backspace_rounded,
                  color: AxonColors.textSecondary,
                  size: 16,
                ),
                const SizedBox(width: 8),
                Text(
                  'Back to study session',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ).animate().fadeIn(delay: 700.ms, duration: 500.ms),

        const SizedBox(height: 24),
      ],
    );
  }
}

class _AnimatedLockIcon extends StatelessWidget {
  final AnimationController pulseController;
  final AnimationController shimmerController;

  const _AnimatedLockIcon({
    required this.pulseController,
    required this.shimmerController,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Glow rings
        ...List.generate(3, (i) {
          return AnimatedBuilder(
            animation: pulseController,
            builder: (context, child) {
              return Container(
                width: 130 + (i * 28),
                height: 130 + (i * 28),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: AxonColors.accent.withValues(
                      alpha: 0.12 - (i * 0.03),
                    ),
                    width: 1,
                  ),
                ),
              ).animate(onPlay: (c) => c.repeat(reverse: true)).scale(
                    begin: const Offset(0.97, 0.97),
                    end: const Offset(1.03, 1.03),
                    duration: Duration(milliseconds: 1800 + (i * 300)),
                  );
            },
          );
        }),
        // Main icon container
        AnimatedBuilder(
          animation: pulseController,
          builder: (context, child) {
            return Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    AxonColors.accent.withValues(alpha: 0.35 * pulseController.value),
                    AxonColors.accent.withValues(alpha: 0.08),
                  ],
                ),
                border: Border.all(
                  color: AxonColors.accent.withValues(alpha: 0.5),
                  width: 1.5,
                ),
                boxShadow: [
                  BoxShadow(
                    color: AxonColors.accent.withValues(alpha: 0.3),
                    blurRadius: 30,
                    spreadRadius: pulseController.value * 3,
                  ),
                ],
              ),
              child: Icon(
                Icons.lock_rounded,
                size: 44,
                color: Colors.white,
              ),
            );
          },
        ),
        // Shimmer overlay
        AnimatedBuilder(
          animation: shimmerController,
          builder: (context, child) {
            return Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  begin: Alignment(-1 + shimmerController.value * 2, -1),
                  end: Alignment(1 - shimmerController.value * 2, 1),
                  colors: [
                    Colors.transparent,
                    Colors.white.withValues(alpha: 0.08),
                    Colors.transparent,
                  ],
                ),
              ),
            );
          },
        ),
      ],
    );
  }
}

class _ProgressCard extends StatelessWidget {
  final AnimationController pulseController;
  final AnimationController orbitController;

  const _ProgressCard({
    required this.pulseController,
    required this.orbitController,
  });

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: StudyLockService.instance.getStatus(),
      builder: (context, snapshot) {
        final logged = snapshot.data?['loggedMinutes'] ?? 0;
        final required = snapshot.data?['requiredMinutes'] ?? 60;
        final progress = required > 0 ? (logged / required).clamp(0.0, 1.0) : 0.0;
        final remaining = (required - logged).clamp(0, required);

        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 24, sigmaY: 24),
              child: Container(
                padding: const EdgeInsets.all(28),
                decoration: BoxDecoration(
                  color: AxonColors.surface.withValues(alpha: 0.06),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.06),
                  ),
                ),
                child: Column(
                  children: [
                    SizedBox(
                      width: 140,
                      height: 140,
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          // Background ring
                          CustomPaint(
                            size: const Size(140, 140),
                            painter: _RingPainter(
                              progress: 1.0,
                              color: Colors.white.withValues(alpha: 0.06),
                              strokeWidth: 6,
                            ),
                          ),
                          // Progress ring
                          AnimatedBuilder(
                            animation: pulseController,
                            builder: (context, child) {
                              return CustomPaint(
                                size: const Size(140, 140),
                                painter: _RingPainter(
                                  progress: progress,
                                  color: AxonColors.accent,
                                  strokeWidth: 6,
                                  glowRadius: pulseController.value * 3,
                                ),
                              );
                            },
                          ),
                          // Center text
                          Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                '$remaining',
                                style: GoogleFonts.orbitron(
                                  fontSize: 36,
                                  fontWeight: FontWeight.w800,
                                  color: Colors.white,
                                  height: 1,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'min left',
                                style: GoogleFonts.googleSans(
                                  fontSize: 12,
                                  color: AxonColors.textSecondary,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      '$logged of $required minutes completed',
                      style: GoogleFonts.googleSans(
                        fontSize: 12,
                        color: AxonColors.textTertiary,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _RingPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double strokeWidth;
  final double glowRadius;

  _RingPainter({
    required this.progress,
    required this.color,
    required this.strokeWidth,
    this.glowRadius = 0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    if (glowRadius > 0) {
      final glowPaint = Paint()
        ..color = color.withValues(alpha: 0.25)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth + glowRadius
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
      canvas.drawCircle(center, radius, glowPaint);
    }

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final startAngle = -math.pi / 2;
    final sweepAngle = 2 * math.pi * progress;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _RingPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.color != color ||
        oldDelegate.glowRadius != glowRadius;
  }
}

class _StatsRow extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 48),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _StatItem(
            icon: Icons.timer_outlined,
            value: 'Focus',
            label: 'Mode',
          ),
          Container(
            width: 1,
            height: 28,
            color: Colors.white.withValues(alpha: 0.08),
          ),
          _StatItem(
            icon: Icons.check_circle_outline,
            value: 'Locked',
            label: 'Until Done',
          ),
          Container(
            width: 1,
            height: 28,
            color: Colors.white.withValues(alpha: 0.08),
          ),
          _StatItem(
            icon: Icons.auto_awesome,
            value: 'Stay',
            label: 'Strong',
          ),
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final IconData icon;
  final String value;
  final String label;

  const _StatItem({
    required this.icon,
    required this.value,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: AxonColors.accent, size: 20),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.googleSans(
            color: AxonColors.textTertiary,
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}

final blockedAppProvider =
    StateProvider<({String appName, String packageName})?>((ref) => null);

class BlockedAppOverlay {
  static void init(WidgetRef ref) {
    StudyLockService.instance.onBlockedAppDetected = (appName, packageName) {
      ref.read(blockedAppProvider.notifier).state =
          (appName: appName, packageName: packageName);
    };
  }

  static void dismiss(WidgetRef ref) {
    ref.read(blockedAppProvider.notifier).state = null;
  }
}
