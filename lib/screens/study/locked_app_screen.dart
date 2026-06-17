// lib/screens/study/locked_app_screen.dart
import 'dart:math' as math;
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
  late AnimationController _particleController;
  late AnimationController _breathController;

  late Animation<double> _pulseAnimation;
  late Animation<double> _orbitAnimation;
  late Animation<double> _shimmerAnimation;
  late Animation<double> _breathAnimation;

  @override
  void initState() {
    super.initState();

    // Pulse animation for lock icon
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.9, end: 1.1).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    // Orbit animation for particles
    _orbitController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 4000),
    )..repeat();
    _orbitAnimation = Tween<double>(begin: 0, end: 2 * math.pi).animate(
      CurvedAnimation(parent: _orbitController, curve: Curves.linear),
    );

    // Shimmer animation for glow effects
    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();
    _shimmerAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _shimmerController, curve: Curves.easeInOut),
    );

    // Particle animation
    _particleController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 6000),
    )..repeat();
    _breathController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 4000),
    )..repeat(reverse: true);
    _breathAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _breathController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _orbitController.dispose();
    _shimmerController.dispose();
    _particleController.dispose();
    _breathController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: AxonGradients.backgroundGradient,
        ),
        child: Stack(
          children: [
            // Animated background particles
            _AnimatedBackground(
              orbitAnimation: _orbitAnimation,
              breathAnimation: _breathAnimation,
              particleController: _particleController,
            ),

            // Main content
            SafeArea(
              child: Column(
                children: [
                  const Spacer(flex: 1),

                  // Lock icon with animations
                  _AnimatedLockIcon(
                    pulseAnimation: _pulseAnimation,
                    shimmerAnimation: _shimmerAnimation,
                    appName: widget.appName,
                  ),

                  const SizedBox(height: 48),

                  // Title
                  _buildTitle(),

                  const SizedBox(height: 20),

                  // Subtitle message
                  _buildSubtitle(),

                  const SizedBox(height: 48),

                  // Progress indicator
                  _ProgressRing(
                    orbitAnimation: _orbitAnimation,
                    pulseAnimation: _pulseAnimation,
                  ),

                  const SizedBox(height: 32),

                  // Stats row
                  _StatsRow(),

                  const Spacer(flex: 2),

                  // Motivational quote
                  _MotivationalQuote(),

                  const SizedBox(height: 40),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTitle() {
    return Text(
      'Stay Focused',
      style: GoogleFonts.orbitron(
        fontSize: 32,
        fontWeight: FontWeight.w800,
        color: Colors.white,
        letterSpacing: 2,
      ),
    )
        .animate()
        .fadeIn(duration: 800.ms, curve: Curves.easeOut)
        .scale(
          begin: const Offset(0.8, 0.8),
          end: const Offset(1, 1),
          duration: 800.ms,
          curve: Curves.easeOutBack,
        )
        .then()
        .shimmer(
          delay: 1200.ms,
          duration: 2000.ms,
          color: const Color(0xFF3A86FF).withValues(alpha: 0.3),
        );
  }

  Widget _buildSubtitle() {
    return Column(
      children: [
        Text(
          '${widget.appName} is locked until you',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            color: Colors.grey[400],
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'complete your study session',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            color: Colors.grey[400],
          ),
        ),
      ],
    )
        .animate()
        .fadeIn(delay: 400.ms, duration: 600.ms)
        .slideY(begin: 0.2, end: 0, delay: 400.ms, duration: 600.ms);
  }
}

class _AnimatedBackground extends StatelessWidget {
  final Animation<double> orbitAnimation;
  final Animation<double> breathAnimation;
  final AnimationController particleController;

  const _AnimatedBackground({
    required this.orbitAnimation,
    required this.breathAnimation,
    required this.particleController,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge(
          [orbitAnimation, breathAnimation, particleController]),
      builder: (context, child) {
        return CustomPaint(
          painter: _ParticlePainter(
            orbitValue: orbitAnimation.value,
            breathValue: breathAnimation.value,
            particleValue: particleController.value,
          ),
          size: Size.infinite,
        );
      },
    );
  }
}

class _ParticlePainter extends CustomPainter {
  final double orbitValue;
  final double breathValue;
  final double particleValue;

  _ParticlePainter({
    required this.orbitValue,
    required this.breathValue,
    required this.particleValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);

    // Draw orbit rings
    for (int i = 0; i < 3; i++) {
      final radius = (size.width * 0.3) + (i * 40) * breathValue;
      final paint = Paint()
        ..color = const Color(0xFF3A86FF).withValues(alpha: 0.05 + (i * 0.02))
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1;

      canvas.drawCircle(center, radius, paint);
    }

    // Draw orbiting particles
    final particlePaint = Paint()
      ..color = const Color(0xFF3A86FF).withValues(alpha: 0.6)
      ..style = PaintingStyle.fill;

    for (int i = 0; i < 8; i++) {
      final angle = orbitValue + (i * math.pi / 4);
      final radius = size.width * 0.35;
      final x = center.dx + radius * math.cos(angle);
      final y = center.dy + radius * math.sin(angle);

      // Particle glow
      final glowPaint = Paint()
        ..color = const Color(0xFF3A86FF).withValues(alpha: 0.2)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
      canvas.drawCircle(Offset(x, y), 6, glowPaint);
      canvas.drawCircle(Offset(x, y), 3, particlePaint);
    }

    // Draw floating particles
    for (int i = 0; i < 12; i++) {
      final baseAngle = (i / 12) * 2 * math.pi;
      final drift = math.sin(particleValue * 2 * math.pi + i) * 20;
      final radius = size.width * (0.4 + (i % 3) * 0.1) + drift;
      final angle = baseAngle + orbitValue * 0.5;

      final x = center.dx + radius * math.cos(angle);
      final y = center.dy + radius * math.sin(angle) * breathValue;

      final floatingPaint = Paint()
        ..color = Colors.white.withValues(alpha: 0.1 + (i % 4) * 0.05)
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(x, y), 2 + (i % 3), floatingPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _ParticlePainter oldDelegate) {
    return oldDelegate.orbitValue != orbitValue ||
        oldDelegate.breathValue != breathValue ||
        oldDelegate.particleValue != particleValue;
  }
}

class _AnimatedLockIcon extends StatelessWidget {
  final Animation<double> pulseAnimation;
  final Animation<double> shimmerAnimation;
  final String appName;

  const _AnimatedLockIcon({
    required this.pulseAnimation,
    required this.shimmerAnimation,
    required this.appName,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Outer glow rings
        _GlowRing(delay: 0, pulseAnimation: pulseAnimation),
        _GlowRing(delay: 200, pulseAnimation: pulseAnimation),
        _GlowRing(delay: 400, pulseAnimation: pulseAnimation),

        // Main lock container
        Container(
          width: 140,
          height: 140,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: RadialGradient(
              colors: [
                const Color(0xFF3A86FF).withValues(alpha: 0.3),
                const Color(0xFF3A86FF).withValues(alpha: 0.1),
                Colors.transparent,
              ],
            ),
          ),
          child: Center(
            child: Transform.scale(
              scale: pulseAnimation.value,
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      const Color(0xFF3A86FF).withValues(alpha: 0.4),
                      const Color(0xFF3A86FF).withValues(alpha: 0.15),
                    ],
                  ),
                  border: Border.all(
                    color: const Color(0xFF3A86FF).withValues(alpha: 0.5),
                    width: 2,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xFF3A86FF).withValues(alpha: 0.4),
                      blurRadius: 30,
                      spreadRadius: 5,
                    ),
                  ],
                ),
                child: Icon(
                  Icons.lock_rounded,
                  size: 50,
                  color: Colors.white,
                )
                    .animate(
                      onPlay: (c) => c.repeat(reverse: true),
                    )
                    .scale(
                      begin: const Offset(0.95, 0.95),
                      end: const Offset(1.05, 1.05),
                      duration: 1500.ms,
                    )
                    .then()
                    .shimmer(
                      delay: 500.ms,
                      duration: 1000.ms,
                      color: Colors.white24,
                    ),
              ),
            ),
          ),
        ).animate().fadeIn(duration: 1000.ms).scale(
              begin: const Offset(0.5, 0.5),
              end: const Offset(1, 1),
              duration: 1000.ms,
              curve: Curves.easeOutBack,
            ),

        // App name badge
        Positioned(
          bottom: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFF3A86FF).withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.4),
              ),
            ),
            child: Text(
              appName,
              style: GoogleFonts.googleSans(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          )
              .animate()
              .fadeIn(delay: 600.ms, duration: 400.ms)
              .slideY(begin: 0.5, end: 0, delay: 600.ms, duration: 400.ms),
        ),
      ],
    );
  }
}

class _GlowRing extends StatelessWidget {
  final int delay;
  final Animation<double> pulseAnimation;

  const _GlowRing({required this.delay, required this.pulseAnimation});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: pulseAnimation,
      builder: (context, child) {
        return Container(
          width: 160 + (delay ~/ 2).toDouble(),
          height: 160 + (delay ~/ 2).toDouble(),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(
              color: const Color(0xFF3A86FF)
                  .withValues(alpha: 0.2 - (delay / 500).toDouble()),
              width: 1,
            ),
          ),
        );
      },
    );
  }
}

class _ProgressRing extends StatelessWidget {
  final Animation<double> orbitAnimation;
  final Animation<double> pulseAnimation;

  const _ProgressRing({
    required this.orbitAnimation,
    required this.pulseAnimation,
  });

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: StudyLockService.instance.getStatus(),
      builder: (context, snapshot) {
        final logged = snapshot.data?['loggedMinutes'] ?? 0;
        final required = snapshot.data?['requiredMinutes'] ?? 60;
        final progress =
            required > 0 ? (logged / required).clamp(0.0, 1.0) : 0.0;
        final remaining = (required - logged).clamp(0, required);

        return Column(
          children: [
            SizedBox(
              width: 160,
              height: 160,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Background ring
                  CustomPaint(
                    size: const Size(160, 160),
                    painter: _RingPainter(
                      progress: 1.0,
                      color: Colors.grey[900]!,
                      strokeWidth: 8,
                    ),
                  ),
                  // Progress ring with animation
                  AnimatedBuilder(
                    animation:
                        Listenable.merge([orbitAnimation, pulseAnimation]),
                    builder: (context, child) {
                      return Transform.rotate(
                        angle: orbitAnimation.value * 0.5,
                        child: CustomPaint(
                          size: const Size(160, 160),
                          painter: _RingPainter(
                            progress: progress,
                            color: const Color(0xFF3A86FF),
                            strokeWidth: 8,
                            glowRadius: pulseAnimation.value * 5,
                          ),
                        ),
                      );
                    },
                  ),
                  // Center content
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '$remaining',
                        style: GoogleFonts.orbitron(
                          fontSize: 40,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ).animate().fadeIn(delay: 800.ms, duration: 400.ms).scale(
                            begin: const Offset(0.8, 0.8),
                            end: const Offset(1, 1),
                            delay: 800.ms,
                            duration: 400.ms,
                          ),
                      Text(
                        'min left',
                        style: GoogleFonts.googleSans(
                          fontSize: 14,
                          color: Colors.grey[500],
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ).animate().fadeIn(delay: 600.ms, duration: 800.ms).scale(
                  begin: const Offset(0.8, 0.8),
                  end: const Offset(1, 1),
                  delay: 600.ms,
                  duration: 600.ms,
                  curve: Curves.easeOutBack,
                ),
            const SizedBox(height: 16),
            Text(
              '$logged of $required minutes completed',
              style: GoogleFonts.googleSans(
                fontSize: 13,
                color: Colors.grey[500],
              ),
            ).animate().fadeIn(delay: 900.ms, duration: 400.ms),
          ],
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

    // Glow effect
    if (glowRadius > 0) {
      final glowPaint = Paint()
        ..color = color.withValues(alpha: 0.3)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth + glowRadius
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, glowRadius);
      canvas.drawCircle(center, radius, glowPaint);
    }

    // Main ring
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
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _StatItem(
            icon: Icons.timer_outlined,
            value: 'Focus',
            label: 'Mode',
          )
              .animate()
              .fadeIn(delay: 1000.ms, duration: 400.ms)
              .slideX(begin: -0.2, end: 0, delay: 1000.ms, duration: 400.ms),
          Container(
            width: 1,
            height: 40,
            color: Colors.grey[800],
          ),
          _StatItem(
            icon: Icons.check_circle_outline,
            value: 'Locked',
            label: 'Until Done',
          )
              .animate()
              .fadeIn(delay: 1100.ms, duration: 400.ms)
              .slideY(begin: 0.2, end: 0, delay: 1100.ms, duration: 400.ms),
          Container(
            width: 1,
            height: 40,
            color: Colors.grey[800],
          ),
          _StatItem(
            icon: Icons.auto_awesome,
            value: 'Stay',
            label: 'Strong',
          )
              .animate()
              .fadeIn(delay: 1200.ms, duration: 400.ms)
              .slideX(begin: 0.2, end: 0, delay: 1200.ms, duration: 400.ms),
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
        Icon(icon, color: const Color(0xFF3A86FF), size: 24),
        const SizedBox(height: 6),
        Text(
          value,
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.googleSans(
            color: Colors.grey[600],
            fontSize: 11,
          ),
        ),
      ],
    );
  }
}

class _MotivationalQuote extends StatelessWidget {
  static const _quotes = [
    '"Focus is the key to success. Every minute spent here builds your future."',
    '"Discipline is the bridge between goals and accomplishments."',
    '"The secret of getting ahead is getting started."',
    '"Your future self will thank you for staying focused today."',
    '"Small daily improvements are the key to staggering long-term results."',
  ];

  @override
  Widget build(BuildContext context) {
    final quote = _quotes[DateTime.now().minute % _quotes.length];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Text(
        quote,
        style: GoogleFonts.googleSans(
          fontSize: 13,
          color: Colors.grey[600],
          fontStyle: FontStyle.italic,
          height: 1.5,
        ),
        textAlign: TextAlign.center,
      ),
    ).animate().fadeIn(delay: 1400.ms, duration: 800.ms).then().shimmer(
          delay: 3000.ms,
          duration: 2000.ms,
          color: const Color(0xFF3A86FF).withValues(alpha: 0.1),
        );
  }
}

// Riverpod provider for blocked app overlay state
final blockedAppProvider =
    StateProvider<({String appName, String packageName})?>((ref) => null);

// Manages blocked app overlay across the app
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
